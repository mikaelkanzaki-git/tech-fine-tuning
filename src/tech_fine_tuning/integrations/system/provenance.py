"""Resolução da revisão exata que produziu uma execução."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path

from tech_fine_tuning.errors import TrainingPreflightError

_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _git_output(arguments: list[str], *, cwd: Path) -> str | None:
    git_executable = shutil.which("git")
    if git_executable is None:
        return None
    try:
        completed = subprocess.run(  # noqa: S603
            [git_executable, *arguments],
            cwd=cwd,
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def resolve_producer_commit(
    *,
    project_root: Path,
    explicit: str | None = None,
    allow_dirty: bool = False,
    environment: Mapping[str, str] | None = None,
) -> str:
    """Usa valor explícito/ambiente em contêiner e Git no desenvolvimento local."""

    values = environment if environment is not None else os.environ
    candidate = explicit or values.get("TECH_FINE_TUNING_REVISION")
    if candidate is not None:
        candidate = candidate.strip().lower()
        if not _COMMIT.fullmatch(candidate):
            raise TrainingPreflightError(
                "TECH_FINE_TUNING_REVISION deve conter um commit Git completo de 40 caracteres."
            )
        return candidate

    commit = _git_output(["rev-parse", "HEAD"], cwd=project_root)
    if commit is None or not _COMMIT.fullmatch(commit):
        raise TrainingPreflightError(
            "Não foi possível determinar o commit produtor; informe --producer-commit."
        )
    dirty = _git_output(["status", "--porcelain"], cwd=project_root)
    if dirty is None:
        raise TrainingPreflightError("Não foi possível verificar o estado do repositório Git.")
    if dirty and not allow_dirty:
        raise TrainingPreflightError(
            "O repositório possui mudanças não commitadas; use --allow-dirty somente para testes."
        )
    return commit
