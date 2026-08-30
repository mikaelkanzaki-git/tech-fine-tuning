"""Diagnóstico serializável do ambiente de treinamento."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class GpuDiagnostic:
    """GPU visível ao processo e memória anunciada pelo driver."""

    name: str
    memory_total_mib: int
    driver_version: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "memory_total_mib": self.memory_total_mib,
            "driver_version": self.driver_version,
        }


@dataclass(frozen=True, slots=True)
class EnvironmentDiagnostic:
    """Capacidades necessárias para reproduzir ou bloquear uma execução."""

    platform: str
    python_version: str
    python_executable: str
    packages: dict[str, str | None]
    gpus: tuple[GpuDiagnostic, ...]
    cuda_available: bool
    cuda_runtime_version: str | None
    bf16_supported: bool
    ready_for_training: bool
    issues: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "python": {
                "version": self.python_version,
                "executable": self.python_executable,
            },
            "packages": self.packages,
            "gpus": [gpu.as_dict() for gpu in self.gpus],
            "cuda": {
                "available": self.cuda_available,
                "runtime_version": self.cuda_runtime_version,
                "bf16_supported": self.bf16_supported,
            },
            "ready_for_training": self.ready_for_training,
            "issues": list(self.issues),
        }
