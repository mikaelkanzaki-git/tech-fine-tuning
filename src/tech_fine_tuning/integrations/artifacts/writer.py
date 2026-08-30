"""Escrita determinística de manifestos de treinamento."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tech_fine_tuning.errors import TrainingExecutionError


def write_artifact_manifest(path: Path, data: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        raise TrainingExecutionError(f"Não foi possível escrever {path}: {error}") from error
