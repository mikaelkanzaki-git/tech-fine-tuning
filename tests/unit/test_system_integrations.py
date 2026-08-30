from __future__ import annotations

import importlib
import shutil
import subprocess
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import pytest

from tech_fine_tuning.errors import TrainingExecutionError, TrainingPreflightError
from tech_fine_tuning.integrations.artifacts.writer import write_artifact_manifest
from tech_fine_tuning.integrations.system import diagnostics, provenance


def test_package_versions_reports_installed_and_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_version(package: str) -> str:
        if package == "torch":
            return "2.0"
        raise metadata.PackageNotFoundError

    monkeypatch.setattr(metadata, "version", fake_version)

    versions = diagnostics._package_versions()

    assert versions["torch"] == "2.0"
    assert versions["unsloth"] is None


def test_visible_gpus_parses_valid_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    completed = SimpleNamespace(
        returncode=0,
        stdout="RTX 5060 Ti, 16311, 591.44\ninvalid\nGPU, nope, 1\n",
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: completed)

    result = diagnostics._visible_gpus()

    assert len(result) == 1
    assert result[0].memory_total_mib == 16311


def test_visible_gpus_tolerates_command_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout=""),
    )
    assert diagnostics._visible_gpus() == ()

    def timeout(*args: object, **kwargs: object) -> object:
        raise subprocess.TimeoutExpired("nvidia-smi", 10)

    monkeypatch.setattr(subprocess, "run", timeout)
    assert diagnostics._visible_gpus() == ()


def test_torch_cuda_reports_capabilities_and_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cuda = SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True)
    torch = SimpleNamespace(cuda=cuda, version=SimpleNamespace(cuda="13.0"))
    monkeypatch.setattr(importlib, "import_module", lambda name: torch)
    assert diagnostics._torch_cuda() == (True, "13.0", True)

    def missing(name: str) -> object:
        raise ImportError(name)

    monkeypatch.setattr(importlib, "import_module", missing)
    assert diagnostics._torch_cuda() == (False, None, False)


def test_environment_diagnostic_explains_unready_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        diagnostics,
        "_package_versions",
        lambda: {
            package: ("1" if package == "torch" else None)
            for package in diagnostics._TRAINING_PACKAGES
        },
    )
    monkeypatch.setattr(diagnostics, "_visible_gpus", lambda: ())
    monkeypatch.setattr(diagnostics, "_torch_cuda", lambda: (False, None, False))

    result = diagnostics.collect_environment_diagnostic()

    assert result.ready_for_training is False
    assert "Nenhuma GPU NVIDIA" in result.issues[0]
    assert any("unsloth" in issue for issue in result.issues)
    assert result.as_dict()["cuda"]["available"] is False


def test_provenance_accepts_explicit_commit_and_rejects_invalid() -> None:
    assert provenance.resolve_producer_commit(
        project_root=Path.cwd(), explicit="A" * 40, environment={}
    ) == "a" * 40
    with pytest.raises(TrainingPreflightError, match="40 caracteres"):
        provenance.resolve_producer_commit(
            project_root=Path.cwd(), explicit="main", environment={}
        )


def test_provenance_uses_clean_git_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    outputs = iter(["b" * 40, ""])
    monkeypatch.setattr(provenance, "_git_output", lambda arguments, cwd: next(outputs))

    assert provenance.resolve_producer_commit(
        project_root=Path.cwd(), environment={}
    ) == "b" * 40

    outputs = iter(["b" * 40, "changed"])
    monkeypatch.setattr(provenance, "_git_output", lambda arguments, cwd: next(outputs))
    with pytest.raises(TrainingPreflightError, match="mudanças não commitadas"):
        provenance.resolve_producer_commit(project_root=Path.cwd(), environment={})


def test_git_output_handles_missing_git_and_process_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert provenance._git_output(["status"], cwd=Path.cwd()) is None

    monkeypatch.setattr(shutil, "which", lambda name: "git.exe")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=" value \n"),
    )
    assert provenance._git_output(["status"], cwd=Path.cwd()) == "value"


def test_artifact_writer_round_trip_and_reports_os_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "nested" / "manifest.json"
    write_artifact_manifest(path, {"z": 1, "a": 2})
    assert path.read_text(encoding="utf-8").startswith('{\n  "a"')

    def fail_write(*args: object, **kwargs: object) -> int:
        raise OSError("disk")

    monkeypatch.setattr(Path, "write_text", fail_write)
    with pytest.raises(TrainingExecutionError, match="Não foi possível escrever"):
        write_artifact_manifest(tmp_path / "error.json", {})
