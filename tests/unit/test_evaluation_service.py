from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from tech_fine_tuning.errors import EvaluationExecutionError, EvaluationPreflightError
from tech_fine_tuning.integrations.dataset.jsonl import file_sha256, write_json
from tech_fine_tuning.models.evaluation import GeneratedComparison
from tech_fine_tuning.services.evaluation_service import (
    build_evaluation_plan,
    execute_evaluation,
)
from tech_fine_tuning.services.sft_dataset_service import prepare_sft_dataset

from .test_sft_dataset_service import canonical_record, create_source


def _evaluation_fixture(tmp_path: Path) -> tuple[Path, Path]:
    records = {
        "train": [canonical_record(1), canonical_record(2)],
        "validation": [canonical_record(3), canonical_record(4), canonical_record(5)],
        "test": [canonical_record(6)],
    }
    dataset_path = tmp_path / "sft"
    prepare_sft_dataset(
        create_source(tmp_path, records_by_split=records),
        dataset_path,
        system_prompt="Safe medical assistant",
    )
    run_path = tmp_path / "run"
    adapter_path = run_path / "adapter"
    adapter_path.mkdir(parents=True)
    write_json(adapter_path / "adapter_config.json", {"peft_type": "LORA"})
    (adapter_path / "adapter_model.safetensors").write_bytes(b"weights")
    write_json(
        run_path / "run-manifest.json",
        {
            "schema_version": "1.0",
            "status": "completed",
            "plan": {
                "config": {
                    "model": {
                        "id": "unsloth/model-bnb-4bit",
                        "revision": "a" * 40,
                        "chat_template": "qwen3-instruct",
                        "max_sequence_length": 2048,
                        "load_in_4bit": True,
                    }
                }
            },
        },
    )
    model_manifest_path = run_path / "model-manifest.json"
    write_json(
        model_manifest_path,
        {
            "schema_version": "1.0",
            "artifact_type": "peft_adapter",
            "artifact": {"uri": "adapter"},
            "provenance": {
                "dataset_manifest_sha256": file_sha256(dataset_path / "manifest.json")
            },
        },
    )
    return model_manifest_path, dataset_path


def _plan(tmp_path: Path, *, sample_size: int = 2, compare_base: bool = True) -> Any:
    model_manifest_path, dataset_path = _evaluation_fixture(tmp_path)
    return build_evaluation_plan(
        model_manifest_path=model_manifest_path,
        dataset_path=dataset_path,
        output_path=tmp_path / "evaluation",
        split="validation",
        sample_size=sample_size,
        seed=3407,
        compare_base=compare_base,
        max_new_tokens=64,
    )


def test_plan_validates_provenance_and_selects_a_deterministic_sample(tmp_path: Path) -> None:
    plan = _plan(tmp_path)

    assert plan.model_id == "unsloth/model-bnb-4bit"
    assert plan.load_in_4bit is True
    assert plan.do_sample is True
    assert plan.temperature == 0.7
    assert plan.top_p == 0.8
    assert plan.top_k == 20
    assert plan.repetition_penalty == 1.1
    assert plan.adapter_path.name == "adapter"
    assert len(plan.examples) == 2
    assert plan.as_dict()["record_ids"] == [example.record_id for example in plan.examples]
    assert plan.as_dict()["evaluation"]["generation"] == {
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1.1,
    }


@pytest.mark.parametrize(
    "generation_options",
    [
        {"temperature": 0},
        {"top_p": 0},
        {"top_p": 1.1},
        {"top_k": 0},
        {"repetition_penalty": 0.9},
    ],
)
def test_plan_rejects_invalid_generation_parameters(
    tmp_path: Path,
    generation_options: dict[str, Any],
) -> None:
    model_manifest_path, dataset_path = _evaluation_fixture(tmp_path)

    with pytest.raises(EvaluationPreflightError, match="Parâmetros de geração inválidos"):
        build_evaluation_plan(
            model_manifest_path=model_manifest_path,
            dataset_path=dataset_path,
            output_path=tmp_path / "evaluation",
            split="validation",
            sample_size=1,
            seed=1,
            compare_base=True,
            max_new_tokens=8,
            **generation_options,
        )


def test_evaluation_writes_comparison_metrics_and_human_review(tmp_path: Path) -> None:
    model_manifest_path, dataset_path = _evaluation_fixture(tmp_path)
    output_path = tmp_path / "evaluation"

    def backend(plan: Any) -> tuple[GeneratedComparison, ...]:
        return tuple(
            GeneratedComparison(
                record_id=example.record_id,
                fine_tuned_answer=example.reference_answer,
                fine_tuned_latency_seconds=0.2,
                base_answer="Unrelated response",
                base_latency_seconds=0.1,
            )
            for example in plan.examples
        )

    outcome = execute_evaluation(
        model_manifest_path=model_manifest_path,
        dataset_path=dataset_path,
        output_path=output_path,
        split="validation",
        sample_size=2,
        seed=3407,
        compare_base=True,
        max_new_tokens=64,
        backend=backend,
    )

    assert outcome.summary["examples"] == 2
    assert outcome.summary["fine_tuned_mean_reference_token_f1"] == 1.0
    assert outcome.summary["token_f1_tuned_wins"] == 2
    assert outcome.responses_path.is_file()
    assert outcome.metrics_path.is_file()
    assert outcome.manifest_path.is_file()
    with outcome.human_review_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert rows[0]["fine_tuned_correctness_0_2"] == ""
    manifest = json.loads(outcome.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["outputs_sha256"]["human_review"]


def test_evaluation_can_run_without_base_comparison(tmp_path: Path) -> None:
    model_manifest_path, dataset_path = _evaluation_fixture(tmp_path)

    def backend(plan: Any) -> tuple[GeneratedComparison, ...]:
        example = plan.examples[0]
        return (
            GeneratedComparison(
                record_id=example.record_id,
                fine_tuned_answer="",
                fine_tuned_latency_seconds=0.2,
                base_answer=None,
                base_latency_seconds=None,
            ),
        )

    outcome = execute_evaluation(
        model_manifest_path=model_manifest_path,
        dataset_path=dataset_path,
        output_path=tmp_path / "evaluation",
        split="test",
        sample_size=1,
        seed=1,
        compare_base=False,
        max_new_tokens=8,
        backend=backend,
    )

    assert outcome.summary["fine_tuned_empty_answers"] == 1
    assert "base_mean_reference_token_f1" not in outcome.summary


def test_plan_rejects_invalid_sample_and_modified_dataset(tmp_path: Path) -> None:
    model_manifest_path, dataset_path = _evaluation_fixture(tmp_path)

    with pytest.raises(EvaluationPreflightError, match="excede"):
        build_evaluation_plan(
            model_manifest_path=model_manifest_path,
            dataset_path=dataset_path,
            output_path=tmp_path / "evaluation",
            split="validation",
            sample_size=4,
            seed=1,
            compare_base=True,
            max_new_tokens=8,
        )

    with (dataset_path / "validation.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(EvaluationPreflightError, match="hash do split"):
        build_evaluation_plan(
            model_manifest_path=model_manifest_path,
            dataset_path=dataset_path,
            output_path=tmp_path / "evaluation",
            split="validation",
            sample_size=1,
            seed=1,
            compare_base=True,
            max_new_tokens=8,
        )


def test_plan_rejects_unsafe_adapter_and_nonempty_output(tmp_path: Path) -> None:
    model_manifest_path, dataset_path = _evaluation_fixture(tmp_path)
    manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    manifest["artifact"]["uri"] = "../adapter"
    write_json(model_manifest_path, manifest)

    with pytest.raises(EvaluationPreflightError, match="não pode sair"):
        build_evaluation_plan(
            model_manifest_path=model_manifest_path,
            dataset_path=dataset_path,
            output_path=tmp_path / "evaluation",
            split="validation",
            sample_size=1,
            seed=1,
            compare_base=True,
            max_new_tokens=8,
        )

    manifest["artifact"]["uri"] = "adapter"
    write_json(model_manifest_path, manifest)
    output_path = tmp_path / "evaluation"
    output_path.mkdir()
    (output_path / "existing.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(EvaluationPreflightError, match="não está vazia"):
        build_evaluation_plan(
            model_manifest_path=model_manifest_path,
            dataset_path=dataset_path,
            output_path=output_path,
            split="validation",
            sample_size=1,
            seed=1,
            compare_base=True,
            max_new_tokens=8,
        )


def test_evaluation_rejects_incomplete_backend_response(tmp_path: Path) -> None:
    model_manifest_path, dataset_path = _evaluation_fixture(tmp_path)

    def missing_response(plan: Any) -> tuple[GeneratedComparison, ...]:
        return ()

    with pytest.raises(EvaluationExecutionError, match="exatamente uma resposta"):
        execute_evaluation(
            model_manifest_path=model_manifest_path,
            dataset_path=dataset_path,
            output_path=tmp_path / "evaluation",
            split="validation",
            sample_size=1,
            seed=1,
            compare_base=True,
            max_new_tokens=8,
            backend=missing_response,
        )


def test_evaluation_requires_base_result_when_requested(tmp_path: Path) -> None:
    model_manifest_path, dataset_path = _evaluation_fixture(tmp_path)

    def no_base(plan: Any) -> tuple[GeneratedComparison, ...]:
        example = plan.examples[0]
        return (
            GeneratedComparison(
                record_id=example.record_id,
                fine_tuned_answer="answer",
                fine_tuned_latency_seconds=0.2,
                base_answer=None,
                base_latency_seconds=None,
            ),
        )

    with pytest.raises(EvaluationExecutionError, match="modelo base"):
        execute_evaluation(
            model_manifest_path=model_manifest_path,
            dataset_path=dataset_path,
            output_path=tmp_path / "evaluation",
            split="validation",
            sample_size=1,
            seed=1,
            compare_base=True,
            max_new_tokens=8,
            backend=no_base,
        )
