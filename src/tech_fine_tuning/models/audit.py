"""Contratos da auditoria de qualidade do dataset SFT."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tech_fine_tuning.models.dataset import SplitName


@dataclass(frozen=True, slots=True)
class SftAuditExample:
    """Projeção de um exemplo necessária para detectar riscos de treinamento."""

    split: SplitName
    record_id: str
    question: str
    normalized_question: str
    normalized_answer: str
    source_relative_path: str
    answer_chars: int
    estimated_total_tokens: int
    has_hpo_boilerplate: bool
    is_resource_navigation: bool
    answer_repeats_question: bool
    maximum_sentence_repetitions: int


@dataclass(frozen=True, slots=True)
class SftAuditOutcome:
    """Arquivos e resumo produzidos por uma auditoria SFT."""

    output_path: Path
    summary_path: Path
    issues_path: Path
    report_path: Path
    manifest_path: Path
    summary: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "output_path": str(self.output_path),
            "summary_path": str(self.summary_path),
            "issues_path": str(self.issues_path),
            "report_path": str(self.report_path),
            "manifest_path": str(self.manifest_path),
            "summary": self.summary,
        }
