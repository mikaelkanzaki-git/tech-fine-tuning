"""Contratos da curadoria determinística do dataset SFT."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class SftCurationOutcome:
    """Arquivos e resumo produzidos pela curadoria."""

    output_path: Path
    manifest_path: Path
    excluded_path: Path
    summary: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "output_path": str(self.output_path),
            "manifest_path": str(self.manifest_path),
            "excluded_path": str(self.excluded_path),
            "summary": self.summary,
        }
