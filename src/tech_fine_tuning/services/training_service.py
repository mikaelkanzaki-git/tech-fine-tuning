"""Orquestração reproduzível do treinamento e da publicação do adaptador."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tech_fine_tuning.errors import TrainingExecutionError, TrainingPreflightError
from tech_fine_tuning.integrations.artifacts.writer import write_artifact_manifest
from tech_fine_tuning.integrations.dataset.jsonl import file_sha256, read_json
from tech_fine_tuning.integrations.system.diagnostics import collect_environment_diagnostic
from tech_fine_tuning.integrations.system.provenance import resolve_producer_commit
from tech_fine_tuning.integrations.training.config_reader import read_training_config
from tech_fine_tuning.integrations.training.unsloth import run_unsloth_training
from tech_fine_tuning.models.environment import EnvironmentDiagnostic
from tech_fine_tuning.models.training import (
    BackendTrainingResult,
    TrainingOutcome,
    TrainingPlan,
)

SFT_SCHEMA_VERSION = "1.0"
Backend = Callable[[TrainingPlan], BackendTrainingResult]
DiagnosticCollector = Callable[[], EnvironmentDiagnostic]


def _mapping(data: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = data.get(field)
    if not isinstance(value, dict):
        raise TrainingPreflightError(f"Manifesto SFT: campo {field!r} deve ser um objeto.")
    return value


def _count(summary: Mapping[str, Any], split: str) -> int:
    values = _mapping(summary, "examples_by_split")
    value = values.get(split)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise TrainingPreflightError(
            f"Manifesto SFT: contagem inválida para o split {split!r}."
        )
    return value


def _validate_dataset(source_path: Path) -> tuple[str, int, int]:
    manifest_path = source_path / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != SFT_SCHEMA_VERSION:
        raise TrainingPreflightError(
            f"Manifesto SFT deve usar schema {SFT_SCHEMA_VERSION!r}."
        )

    summary = _mapping(manifest, "summary")
    hashes = _mapping(manifest, "output_files_sha256")
    for split in ("train", "validation", "test"):
        path = source_path / f"{split}.jsonl"
        expected = hashes.get(split)
        if not isinstance(expected, str) or file_sha256(path) != expected:
            raise TrainingPreflightError(
                f"O hash do split {split!r} diverge do manifesto SFT."
            )
    return file_sha256(manifest_path), _count(summary, "train"), _count(summary, "validation")


def _validate_output(output_path: Path, resume_from_checkpoint: Path | None) -> None:
    if output_path.exists() and any(output_path.iterdir()) and resume_from_checkpoint is None:
        raise TrainingPreflightError(
            f"A saída {output_path} não está vazia; escolha outra saída ou retome um checkpoint."
        )
    if resume_from_checkpoint is not None:
        if not resume_from_checkpoint.exists() or not resume_from_checkpoint.is_dir():
            raise TrainingPreflightError(
                f"Checkpoint para retomada não encontrado: {resume_from_checkpoint}"
            )
        try:
            resume_from_checkpoint.relative_to(output_path)
        except ValueError as error:
            raise TrainingPreflightError(
                "O checkpoint de retomada deve estar dentro do diretório de saída da run."
            ) from error


def build_training_plan(
    *,
    config_path: Path,
    source_path: Path,
    output_path: Path,
    project_root: Path,
    resume_from_checkpoint: Path | None = None,
    producer_commit: str | None = None,
    allow_dirty: bool = False,
) -> TrainingPlan:
    """Valida configuração, dataset, destino e procedência antes de carregar a GPU."""

    config_path = config_path.expanduser().resolve()
    source_path = source_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    project_root = project_root.expanduser().resolve()
    resume_path = (
        resume_from_checkpoint.expanduser().resolve() if resume_from_checkpoint else None
    )
    if not source_path.exists() or not source_path.is_dir():
        raise TrainingPreflightError(f"Dataset SFT não encontrado: {source_path}")
    config = read_training_config(config_path)
    dataset_hash, train_examples, validation_examples = _validate_dataset(source_path)
    _validate_output(output_path, resume_path)
    commit = resolve_producer_commit(
        project_root=project_root,
        explicit=producer_commit,
        allow_dirty=allow_dirty,
    )
    return TrainingPlan(
        config=config,
        config_path=config_path,
        source_path=source_path,
        output_path=output_path,
        train_path=source_path / "train.jsonl",
        validation_path=source_path / "validation.jsonl",
        dataset_manifest_sha256=dataset_hash,
        train_examples=train_examples,
        validation_examples=validation_examples,
        producer_commit=commit,
        resume_from_checkpoint=resume_path,
    )


def _model_manifest(plan: TrainingPlan, result: BackendTrainingResult) -> dict[str, Any]:
    try:
        adapter_uri = str(result.adapter_path.resolve().relative_to(plan.output_path))
    except ValueError as error:
        raise TrainingExecutionError(
            "O backend publicou o adaptador fora do diretório da run."
        ) from error
    return {
        "schema_version": "1.0",
        "artifact_id": plan.config.run_name,
        "artifact_type": "peft_adapter",
        "artifact": {"uri": adapter_uri.replace("\\", "/")},
        "base_model": {
            "id": plan.config.model.base_model_id,
            "revision": plan.config.model.base_model_revision,
        },
        "tokenizer": {
            "id": plan.config.model.tokenizer_id,
            "revision": plan.config.model.tokenizer_revision,
        },
        "provenance": {
            "producer": "tech-fine-tuning",
            "producer_commit": plan.producer_commit,
            "dataset_manifest_sha256": plan.dataset_manifest_sha256,
        },
    }


def execute_training(
    *,
    config_path: Path,
    source_path: Path,
    output_path: Path,
    project_root: Path,
    resume_from_checkpoint: Path | None = None,
    producer_commit: str | None = None,
    allow_dirty: bool = False,
    dry_run: bool = False,
    backend: Backend = run_unsloth_training,
    diagnostic_collector: DiagnosticCollector = collect_environment_diagnostic,
) -> TrainingOutcome:
    """Executa um plano ou apenas valida seus contratos com ``dry_run``."""

    plan = build_training_plan(
        config_path=config_path,
        source_path=source_path,
        output_path=output_path,
        project_root=project_root,
        resume_from_checkpoint=resume_from_checkpoint,
        producer_commit=producer_commit,
        allow_dirty=allow_dirty,
    )
    if dry_run:
        return TrainingOutcome(plan=plan, dry_run=True)

    diagnostic = diagnostic_collector()
    if not diagnostic.ready_for_training:
        issues = " ".join(diagnostic.issues)
        raise TrainingPreflightError(f"Ambiente não está pronto para treinamento. {issues}")

    plan.output_path.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(UTC).isoformat()
    run_manifest_path = plan.output_path / "run-manifest.json"
    run_manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "running",
        "started_at": started_at,
        "completed_at": None,
        "plan": plan.as_dict(),
        "environment": diagnostic.as_dict(),
        "metrics": None,
        "peak_reserved_memory_gib": None,
        "failure_type": None,
    }
    write_artifact_manifest(run_manifest_path, run_manifest)
    try:
        result = backend(plan)
        model_manifest = _model_manifest(plan, result)
        model_manifest_path = plan.output_path / "model-manifest.json"
        write_artifact_manifest(model_manifest_path, model_manifest)
    except Exception as error:
        run_manifest.update(
            {
                "status": "failed",
                "completed_at": datetime.now(UTC).isoformat(),
                "failure_type": type(error).__name__,
            }
        )
        write_artifact_manifest(run_manifest_path, run_manifest)
        raise

    run_manifest.update(
        {
            "status": "completed",
            "completed_at": datetime.now(UTC).isoformat(),
            "metrics": result.metrics,
            "peak_reserved_memory_gib": result.peak_reserved_memory_gib,
        }
    )
    write_artifact_manifest(run_manifest_path, run_manifest)
    return TrainingOutcome(
        plan=plan,
        dry_run=False,
        run_manifest_path=run_manifest_path,
        model_manifest_path=model_manifest_path,
    )
