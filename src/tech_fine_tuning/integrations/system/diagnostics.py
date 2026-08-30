"""Inspeção tolerante a falhas de Python, CUDA e pacotes de treinamento."""

from __future__ import annotations

import importlib
import platform
import subprocess
import sys
from importlib import metadata
from typing import Any

from tech_fine_tuning.models.environment import EnvironmentDiagnostic, GpuDiagnostic

_TRAINING_PACKAGES = (
    "torch",
    "unsloth",
    "transformers",
    "trl",
    "datasets",
    "bitsandbytes",
)


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in _TRAINING_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _visible_gpus() -> tuple[GpuDiagnostic, ...]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(  # noqa: S603
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ()
    if completed.returncode != 0:
        return ()

    gpus: list[GpuDiagnostic] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            continue
        try:
            memory_total_mib = int(fields[1])
        except ValueError:
            continue
        gpus.append(
            GpuDiagnostic(
                name=fields[0],
                memory_total_mib=memory_total_mib,
                driver_version=fields[2],
            )
        )
    return tuple(gpus)


def _torch_cuda() -> tuple[bool, str | None, bool]:
    try:
        torch: Any = importlib.import_module("torch")
    except (ImportError, OSError):
        return False, None, False
    try:
        available = bool(torch.cuda.is_available())
        runtime = str(torch.version.cuda) if torch.version.cuda is not None else None
        bf16_supported = bool(available and torch.cuda.is_bf16_supported())
    except (AttributeError, RuntimeError):
        return False, None, False
    return available, runtime, bf16_supported


def collect_environment_diagnostic() -> EnvironmentDiagnostic:
    """Coleta diagnóstico sem exigir que o extra opcional de treino esteja instalado."""

    packages = _package_versions()
    gpus = _visible_gpus()
    cuda_available, cuda_runtime, bf16_supported = _torch_cuda()
    issues: list[str] = []
    if not gpus:
        issues.append("Nenhuma GPU NVIDIA foi encontrada pelo nvidia-smi.")
    for package in _TRAINING_PACKAGES:
        if packages[package] is None:
            issues.append(f"Dependência de treinamento ausente: {package}.")
    if packages["torch"] is not None and not cuda_available:
        issues.append("PyTorch não encontrou um runtime CUDA utilizável.")

    return EnvironmentDiagnostic(
        platform=platform.platform(),
        python_version=platform.python_version(),
        python_executable=sys.executable,
        packages=packages,
        gpus=gpus,
        cuda_available=cuda_available,
        cuda_runtime_version=cuda_runtime,
        bf16_supported=bf16_supported,
        ready_for_training=not issues,
        issues=tuple(issues),
    )
