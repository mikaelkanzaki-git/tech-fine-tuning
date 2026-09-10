"""Orquestração auditável da comparação entre modelo base e adaptador."""

from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any, cast

from tech_fine_tuning.errors import (
    EvaluationExecutionError,
    EvaluationPreflightError,
    TrainingExecutionError,
)
from tech_fine_tuning.integrations.artifacts.writer import write_artifact_manifest
from tech_fine_tuning.integrations.dataset.jsonl import (
    file_sha256,
    read_json,
    read_jsonl,
    write_jsonl,
)
from tech_fine_tuning.integrations.training.unsloth import run_unsloth_evaluation
from tech_fine_tuning.models.evaluation import (
    EvaluationExample,
    EvaluationOutcome,
    EvaluationPlan,
    EvaluationSplit,
    GeneratedComparison,
)

EvaluationBackend = Callable[[EvaluationPlan], tuple[GeneratedComparison, ...]]
_TOKEN = re.compile(r"\w+", flags=re.UNICODE)


def _mapping(data: Mapping[str, Any], field: str, *, context: str) -> Mapping[str, Any]:
    value = data.get(field)
    if not isinstance(value, dict):
        raise EvaluationPreflightError(f"{context}: campo {field!r} deve ser um objeto.")
    return value


def _string(data: Mapping[str, Any], field: str, *, context: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise EvaluationPreflightError(
            f"{context}: campo {field!r} deve ser uma string não vazia."
        )
    return value.strip()


def _positive_integer(data: Mapping[str, Any], field: str, *, context: str) -> int:
    value = data.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise EvaluationPreflightError(f"{context}: campo {field!r} deve ser positivo.")
    return value


def _boolean(data: Mapping[str, Any], field: str, *, context: str) -> bool:
    value = data.get(field)
    if not isinstance(value, bool):
        raise EvaluationPreflightError(f"{context}: campo {field!r} deve ser booleano.")
    return value


def _adapter_path(manifest_path: Path, manifest: Mapping[str, Any]) -> Path:
    artifact = _mapping(manifest, "artifact", context="manifesto do modelo")
    uri = _string(artifact, "uri", context="manifesto do modelo")
    relative_path = Path(uri)
    if relative_path.is_absolute() or "://" in uri:
        raise EvaluationPreflightError(
            "A avaliação local aceita somente artifact.uri relativo ao manifesto."
        )
    root = manifest_path.parent.resolve()
    resolved = (root / relative_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise EvaluationPreflightError(
            "artifact.uri não pode sair do diretório da run."
        ) from error
    if not (resolved / "adapter_config.json").is_file() or not (
        resolved / "adapter_model.safetensors"
    ).is_file():
        raise EvaluationPreflightError(f"Adaptador PEFT incompleto: {resolved}")
    return resolved


def _evaluation_example(raw: Mapping[str, Any], *, context: str) -> EvaluationExample:
    metadata = _mapping(raw, "metadata", context=context)
    record_id = _string(metadata, "record_id", context=context)
    messages = raw.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise EvaluationPreflightError(
            f"{context}: messages deve conter system, user e assistant."
        )
    parsed: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise EvaluationPreflightError(f"{context}: mensagem inválida.")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise EvaluationPreflightError(f"{context}: role/content inválidos.")
        parsed.append({"role": role, "content": content})
    if [message["role"] for message in parsed] != ["system", "user", "assistant"]:
        raise EvaluationPreflightError(f"{context}: ordem das mensagens inválida.")
    return EvaluationExample(
        record_id=record_id,
        question=parsed[1]["content"],
        reference_answer=parsed[2]["content"],
        prompt_messages=tuple(parsed[:2]),
    )


def _select_examples(
    records: tuple[dict[str, Any], ...], *, sample_size: int, seed: int, split_path: Path
) -> tuple[EvaluationExample, ...]:
    examples = tuple(
        _evaluation_example(raw, context=f"{split_path}, linha {line_number}")
        for line_number, raw in enumerate(records, start=1)
    )
    if sample_size > len(examples):
        raise EvaluationPreflightError(
            f"A amostra solicitada ({sample_size}) excede os {len(examples)} exemplos do split."
        )

    def selection_key(example: EvaluationExample) -> str:
        value = f"{seed}:{example.record_id}".encode()
        return hashlib.sha256(value).hexdigest()

    return tuple(sorted(examples, key=selection_key)[:sample_size])


def build_evaluation_plan(
    *,
    model_manifest_path: Path,
    dataset_path: Path,
    output_path: Path,
    split: EvaluationSplit,
    sample_size: int,
    seed: int,
    compare_base: bool,
    max_new_tokens: int,
    do_sample: bool = True,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    repetition_penalty: float = 1.1,
) -> EvaluationPlan:
    """Valida manifestos, hashes, adaptador e amostra sem carregar a GPU."""

    if split not in {"validation", "test"}:
        raise EvaluationPreflightError("Somente validation ou test podem ser avaliados.")
    if sample_size <= 0 or max_new_tokens <= 0:
        raise EvaluationPreflightError("sample-size e max-new-tokens devem ser positivos.")
    invalid_sampling = (
        temperature <= 0
        or not 0 < top_p <= 1
        or top_k <= 0
        or repetition_penalty < 1
    )
    if invalid_sampling:
        raise EvaluationPreflightError(
            "Parâmetros de geração inválidos: temperature > 0, top-p em (0, 1], "
            "top-k > 0 e repetition-penalty >= 1."
        )
    model_manifest_path = model_manifest_path.expanduser().resolve()
    dataset_path = dataset_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if output_path.exists() and any(output_path.iterdir()):
        raise EvaluationPreflightError(f"A saída de avaliação não está vazia: {output_path}")

    model_manifest = read_json(model_manifest_path)
    if model_manifest.get("schema_version") != "1.0":
        raise EvaluationPreflightError("Manifesto do modelo deve usar schema '1.0'.")
    if model_manifest.get("artifact_type") != "peft_adapter":
        raise EvaluationPreflightError("A avaliação atual aceita somente peft_adapter.")
    adapter_path = _adapter_path(model_manifest_path, model_manifest)

    run_manifest_path = model_manifest_path.parent / "run-manifest.json"
    run_manifest = read_json(run_manifest_path)
    if run_manifest.get("status") != "completed":
        raise EvaluationPreflightError("A run de treinamento não está concluída.")
    plan_data = _mapping(run_manifest, "plan", context="manifesto da run")
    config_data = _mapping(plan_data, "config", context="manifesto da run")
    model_data = _mapping(config_data, "model", context="manifesto da run")

    dataset_manifest_path = dataset_path / "manifest.json"
    dataset_manifest = read_json(dataset_manifest_path)
    provenance = _mapping(model_manifest, "provenance", context="manifesto do modelo")
    expected_dataset_hash = _string(
        provenance,
        "dataset_manifest_sha256",
        context="manifesto do modelo",
    )
    if file_sha256(dataset_manifest_path) != expected_dataset_hash:
        raise EvaluationPreflightError(
            "O dataset SFT não é o mesmo registrado no modelo treinado."
        )
    hashes = _mapping(dataset_manifest, "output_files_sha256", context="manifesto SFT")
    split_path = dataset_path / f"{split}.jsonl"
    expected_split_hash = hashes.get(split)
    if not isinstance(expected_split_hash, str) or file_sha256(split_path) != expected_split_hash:
        raise EvaluationPreflightError(f"O hash do split {split!r} diverge do manifesto SFT.")

    records = read_jsonl(split_path)
    examples = _select_examples(
        records,
        sample_size=sample_size,
        seed=seed,
        split_path=split_path,
    )
    return EvaluationPlan(
        model_manifest_path=model_manifest_path,
        run_manifest_path=run_manifest_path,
        adapter_path=adapter_path,
        dataset_path=dataset_path,
        split_path=split_path,
        output_path=output_path,
        split=split,
        sample_size=sample_size,
        seed=seed,
        compare_base=compare_base,
        model_id=_string(model_data, "id", context="manifesto da run"),
        model_revision=_string(model_data, "revision", context="manifesto da run"),
        chat_template=_string(model_data, "chat_template", context="manifesto da run"),
        max_sequence_length=_positive_integer(
            model_data,
            "max_sequence_length",
            context="manifesto da run",
        ),
        load_in_4bit=_boolean(model_data, "load_in_4bit", context="manifesto da run"),
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        examples=examples,
    )


def _token_f1(reference: str, candidate: str) -> float:
    reference_tokens = _TOKEN.findall(reference.casefold())
    candidate_tokens = _TOKEN.findall(candidate.casefold())
    if not reference_tokens or not candidate_tokens:
        return 0.0
    overlap = sum((Counter(reference_tokens) & Counter(candidate_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(candidate_tokens)
    recall = overlap / len(reference_tokens)
    return round(2 * precision * recall / (precision + recall), 4)


def _response_records(
    plan: EvaluationPlan,
    generated: tuple[GeneratedComparison, ...],
) -> tuple[dict[str, Any], ...]:
    by_id = {comparison.record_id: comparison for comparison in generated}
    if len(by_id) != len(plan.examples) or set(by_id) != {
        example.record_id for example in plan.examples
    }:
        raise EvaluationExecutionError(
            "O backend não retornou exatamente uma resposta para cada exemplo."
        )
    records: list[dict[str, Any]] = []
    for example in plan.examples:
        comparison = by_id[example.record_id]
        if plan.compare_base and (
            comparison.base_answer is None or comparison.base_latency_seconds is None
        ):
            raise EvaluationExecutionError(
                "O backend não retornou a resposta do modelo base solicitada."
            )
        record: dict[str, Any] = {
            "record_id": example.record_id,
            "question": example.question,
            "reference_answer": example.reference_answer,
            "fine_tuned_answer": comparison.fine_tuned_answer,
            "fine_tuned_latency_seconds": comparison.fine_tuned_latency_seconds,
            "fine_tuned_reference_token_f1": _token_f1(
                example.reference_answer, comparison.fine_tuned_answer
            ),
            "base_answer": comparison.base_answer,
            "base_latency_seconds": comparison.base_latency_seconds,
            "base_reference_token_f1": (
                _token_f1(example.reference_answer, comparison.base_answer)
                if comparison.base_answer is not None
                else None
            ),
        }
        records.append(record)
    return tuple(records)


def _summary(records: tuple[dict[str, Any], ...], *, compare_base: bool) -> dict[str, Any]:
    fine_scores = [
        cast(float, record["fine_tuned_reference_token_f1"]) for record in records
    ]
    fine_latencies = [
        cast(float, record["fine_tuned_latency_seconds"]) for record in records
    ]
    summary: dict[str, Any] = {
        "examples": len(records),
        "fine_tuned_empty_answers": sum(
            not str(record["fine_tuned_answer"]).strip() for record in records
        ),
        "fine_tuned_mean_reference_token_f1": round(fmean(fine_scores), 4),
        "fine_tuned_mean_latency_seconds": round(fmean(fine_latencies), 4),
    }
    if compare_base:
        base_scores = [cast(float, record["base_reference_token_f1"]) for record in records]
        base_latencies = [cast(float, record["base_latency_seconds"]) for record in records]
        tuned_wins = sum(
            tuned > base for tuned, base in zip(fine_scores, base_scores, strict=True)
        )
        base_wins = sum(
            base > tuned for tuned, base in zip(fine_scores, base_scores, strict=True)
        )
        summary.update(
            {
                "base_empty_answers": sum(
                    not str(record["base_answer"]).strip() for record in records
                ),
                "base_mean_reference_token_f1": round(fmean(base_scores), 4),
                "base_mean_latency_seconds": round(fmean(base_latencies), 4),
                "token_f1_tuned_wins": tuned_wins,
                "token_f1_base_wins": base_wins,
                "token_f1_ties": len(records) - tuned_wins - base_wins,
            }
        )
    return summary


def _write_human_review(path: Path, records: tuple[dict[str, Any], ...]) -> None:
    fieldnames = [
        "record_id",
        "question",
        "reference_answer",
        "base_answer",
        "fine_tuned_answer",
        "base_correctness_0_2",
        "fine_tuned_correctness_0_2",
        "base_completeness_0_2",
        "fine_tuned_completeness_0_2",
        "base_safety_0_2",
        "fine_tuned_safety_0_2",
        "preference_base_fine_tuned_tie",
        "review_notes",
    ]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for record in records:
                writer.writerow(record)
    except OSError as error:
        raise EvaluationExecutionError(
            f"Não foi possível escrever a planilha de revisão {path}: {error}"
        ) from error


def execute_evaluation(
    *,
    model_manifest_path: Path,
    dataset_path: Path,
    output_path: Path,
    split: EvaluationSplit,
    sample_size: int,
    seed: int,
    compare_base: bool,
    max_new_tokens: int,
    do_sample: bool = True,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    repetition_penalty: float = 1.1,
    backend: EvaluationBackend = run_unsloth_evaluation,
) -> EvaluationOutcome:
    """Gera respostas, métricas auxiliares e planilha de revisão humana."""

    plan = build_evaluation_plan(
        model_manifest_path=model_manifest_path,
        dataset_path=dataset_path,
        output_path=output_path,
        split=split,
        sample_size=sample_size,
        seed=seed,
        compare_base=compare_base,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )
    generated = backend(plan)
    records = _response_records(plan, generated)
    summary = _summary(records, compare_base=compare_base)
    responses_path = plan.output_path / "responses.jsonl"
    metrics_path = plan.output_path / "automatic-metrics.json"
    human_review_path = plan.output_path / "human-review.csv"
    manifest_path = plan.output_path / "evaluation-manifest.json"
    try:
        write_jsonl(responses_path, records)
        write_artifact_manifest(metrics_path, {"schema_version": "1.0", "summary": summary})
        _write_human_review(human_review_path, records)
        manifest = {
            "schema_version": "1.0",
            "status": "completed",
            "generated_at": datetime.now(UTC).isoformat(),
            "plan": plan.as_dict(),
            "inputs_sha256": {
                "model_manifest": file_sha256(plan.model_manifest_path),
                "run_manifest": file_sha256(plan.run_manifest_path),
                "dataset_manifest": file_sha256(plan.dataset_path / "manifest.json"),
                "split": file_sha256(plan.split_path),
            },
            "outputs_sha256": {
                "responses": file_sha256(responses_path),
                "automatic_metrics": file_sha256(metrics_path),
                "human_review": file_sha256(human_review_path),
            },
            "summary": summary,
        }
        write_artifact_manifest(manifest_path, manifest)
    except (OSError, UnicodeError, TrainingExecutionError) as error:
        raise EvaluationExecutionError(str(error)) from error
    return EvaluationOutcome(
        output_path=plan.output_path,
        responses_path=responses_path,
        metrics_path=metrics_path,
        human_review_path=human_review_path,
        manifest_path=manifest_path,
        summary=summary,
    )
