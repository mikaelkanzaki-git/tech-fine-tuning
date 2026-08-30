"""Configuração explícita da preparação do dataset SFT."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from tech_fine_tuning.errors import ConfigurationError

DEFAULT_SYSTEM_PROMPT = (
    "You are a medical information assistant. Provide educational information based on "
    "reliable medical sources. Do not prescribe medication, diagnose a patient, or replace "
    "a qualified healthcare professional."
)


@dataclass(frozen=True, slots=True)
class Settings:
    """Caminhos e instrução padrão, substituíveis por variáveis de ambiente."""

    canonical_dataset_path: Path = Path("../tech-ingestao/artifacts/dataset")
    sft_output_path: Path = Path("artifacts/sft")
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    def __post_init__(self) -> None:
        if not self.system_prompt.strip():
            raise ConfigurationError("TECH_FINE_TUNING_SYSTEM_PROMPT não pode ser vazio.")

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> Settings:
        values = environment if environment is not None else os.environ
        return cls(
            canonical_dataset_path=Path(
                values.get(
                    "TECH_FINE_TUNING_CANONICAL_DATASET_PATH",
                    "../tech-ingestao/artifacts/dataset",
                )
            ),
            sft_output_path=Path(values.get("TECH_FINE_TUNING_SFT_OUTPUT_PATH", "artifacts/sft")),
            system_prompt=values.get("TECH_FINE_TUNING_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
        )
