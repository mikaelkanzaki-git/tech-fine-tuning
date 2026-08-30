from __future__ import annotations

import json
from pathlib import Path

import pytest

from tech_fine_tuning.errors import TrainingExecutionError, TrainingPreflightError
from tech_fine_tuning.integrations.dataset.jsonl import write_json
from tech_fine_tuning.models.environment import EnvironmentDiagnostic, GpuDiagnostic
from tech_fine_tuning.models.training import BackendTrainingResult
from tech_fine_tuning.services.sft_dataset_service import prepare_sft_dataset
from tech_fine_tuning.services.training_service import build_training_plan, execute_training

from .test_sft_dataset_service import create_source

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT_ROOT / "configs" / "qwen3-4b" / "smoke.toml"
COMMIT = "c" * 40


def _sft_dataset(tmp_path: Path) -> Path:
    output = tmp_path / "sft"
    prepare_sft_dataset(create_source(tmp_path), output, system_prompt="Safe")
    return output


def _ready_environment() -> EnvironmentDiagnostic:
    return EnvironmentDiagnostic(
        platform="test",
        python_version="3.12",
        python_executable="python",
        packages={"unsloth": "1"},
        gpus=(GpuDiagnostic(name="GPU", memory_total_mib=16000, driver_version="1"),),
        cuda_available=True,
        cuda_runtime_version="13",
        bf16_supported=True,
        ready_for_training=True,
        issues=(),
    )


def test_dry_run_validates_dataset_without_creating_output(tmp_path: Path) -> None:
    source = _sft_dataset(tmp_path)
    output = tmp_path / "run"

    result = execute_training(
        config_path=CONFIG,
        source_path=source,
        output_path=output,
        project_root=PROJECT_ROOT,
        producer_commit=COMMIT,
        dry_run=True,
    )

    assert result.dry_run is True
    assert result.plan.train_examples == 2
    assert result.plan.validation_examples == 1
    assert result.as_dict()["dataset"]["manifest_sha256"]
    assert not output.exists()


def test_training_writes_run_and_model_manifests(tmp_path: Path) -> None:
    source = _sft_dataset(tmp_path)
    output = tmp_path / "run"

    def backend(plan: object) -> BackendTrainingResult:
        adapter = output / "adapter"
        adapter.mkdir(parents=True)
        (adapter / "adapter.safetensors").write_bytes(b"weights")
        return BackendTrainingResult(
            metrics={"validation": {"eval_loss": 1.2}},
            adapter_path=adapter,
            peak_reserved_memory_gib=3.5,
        )

    result = execute_training(
        config_path=CONFIG,
        source_path=source,
        output_path=output,
        project_root=PROJECT_ROOT,
        producer_commit=COMMIT,
        backend=backend,
        diagnostic_collector=_ready_environment,
    )

    assert result.run_manifest_path is not None
    assert result.model_manifest_path is not None
    run_manifest = json.loads(result.run_manifest_path.read_text(encoding="utf-8"))
    model_manifest = json.loads(result.model_manifest_path.read_text(encoding="utf-8"))
    assert run_manifest["status"] == "completed"
    assert run_manifest["metrics"]["validation"]["eval_loss"] == 1.2
    assert model_manifest["artifact"] == {"uri": "adapter"}
    assert model_manifest["provenance"]["producer_commit"] == COMMIT


def test_training_records_backend_failure(tmp_path: Path) -> None:
    source = _sft_dataset(tmp_path)
    output = tmp_path / "run"

    def backend(plan: object) -> BackendTrainingResult:
        raise TrainingExecutionError("failed")

    with pytest.raises(TrainingExecutionError, match="failed"):
        execute_training(
            config_path=CONFIG,
            source_path=source,
            output_path=output,
            project_root=PROJECT_ROOT,
            producer_commit=COMMIT,
            backend=backend,
            diagnostic_collector=_ready_environment,
        )
    manifest = json.loads((output / "run-manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["failure_type"] == "TrainingExecutionError"


def test_training_rejects_unready_environment(tmp_path: Path) -> None:
    source = _sft_dataset(tmp_path)
    environment = _ready_environment()
    unavailable = EnvironmentDiagnostic(
        platform=environment.platform,
        python_version=environment.python_version,
        python_executable=environment.python_executable,
        packages=environment.packages,
        gpus=(),
        cuda_available=False,
        cuda_runtime_version=None,
        bf16_supported=False,
        ready_for_training=False,
        issues=("CUDA ausente.",),
    )
    with pytest.raises(TrainingPreflightError, match="CUDA ausente"):
        execute_training(
            config_path=CONFIG,
            source_path=source,
            output_path=tmp_path / "run",
            project_root=PROJECT_ROOT,
            producer_commit=COMMIT,
            diagnostic_collector=lambda: unavailable,
        )


def test_plan_rejects_missing_and_tampered_dataset(tmp_path: Path) -> None:
    with pytest.raises(TrainingPreflightError, match="não encontrado"):
        build_training_plan(
            config_path=CONFIG,
            source_path=tmp_path / "missing",
            output_path=tmp_path / "run",
            project_root=PROJECT_ROOT,
            producer_commit=COMMIT,
        )

    source = _sft_dataset(tmp_path)
    (source / "train.jsonl").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(TrainingPreflightError, match="hash"):
        build_training_plan(
            config_path=CONFIG,
            source_path=source,
            output_path=tmp_path / "run",
            project_root=PROJECT_ROOT,
            producer_commit=COMMIT,
        )


def test_plan_rejects_invalid_manifest_and_nonempty_output(tmp_path: Path) -> None:
    source = _sft_dataset(tmp_path)
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "2.0"
    write_json(manifest_path, manifest)
    with pytest.raises(TrainingPreflightError, match="schema"):
        build_training_plan(
            config_path=CONFIG,
            source_path=source,
            output_path=tmp_path / "run",
            project_root=PROJECT_ROOT,
            producer_commit=COMMIT,
        )

    source = _sft_dataset(tmp_path / "second")
    output = tmp_path / "nonempty"
    output.mkdir()
    (output / "file").write_text("x", encoding="utf-8")
    with pytest.raises(TrainingPreflightError, match="não está vazia"):
        build_training_plan(
            config_path=CONFIG,
            source_path=source,
            output_path=output,
            project_root=PROJECT_ROOT,
            producer_commit=COMMIT,
        )


def test_plan_validates_resume_checkpoint_location(tmp_path: Path) -> None:
    source = _sft_dataset(tmp_path)
    output = tmp_path / "run"
    missing = output / "checkpoints" / "checkpoint-10"
    with pytest.raises(TrainingPreflightError, match="não encontrado"):
        build_training_plan(
            config_path=CONFIG,
            source_path=source,
            output_path=output,
            project_root=PROJECT_ROOT,
            producer_commit=COMMIT,
            resume_from_checkpoint=missing,
        )

    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(TrainingPreflightError, match="dentro do diretório"):
        build_training_plan(
            config_path=CONFIG,
            source_path=source,
            output_path=output,
            project_root=PROJECT_ROOT,
            producer_commit=COMMIT,
            resume_from_checkpoint=outside,
        )

    missing.mkdir(parents=True)
    plan = build_training_plan(
        config_path=CONFIG,
        source_path=source,
        output_path=output,
        project_root=PROJECT_ROOT,
        producer_commit=COMMIT,
        resume_from_checkpoint=missing,
    )
    assert plan.resume_from_checkpoint == missing.resolve()


def test_training_rejects_adapter_outside_run(tmp_path: Path) -> None:
    source = _sft_dataset(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()

    def backend(plan: object) -> BackendTrainingResult:
        return BackendTrainingResult(
            metrics={}, adapter_path=outside, peak_reserved_memory_gib=None
        )

    with pytest.raises(TrainingExecutionError, match="fora do diretório"):
        execute_training(
            config_path=CONFIG,
            source_path=source,
            output_path=tmp_path / "run",
            project_root=PROJECT_ROOT,
            producer_commit=COMMIT,
            backend=backend,
            diagnostic_collector=_ready_environment,
        )
