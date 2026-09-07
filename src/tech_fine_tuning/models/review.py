"""Contratos da consolidação da revisão humana."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

ReviewPreference = Literal["base", "fine_tuned", "tie"]
ReviewDecision = Literal["approved", "rejected"]


@dataclass(frozen=True, slots=True)
class HumanReviewRecord:
    """Notas normalizadas de uma comparação entre base e fine-tuned."""

    record_id: str
    base_correctness: int
    fine_tuned_correctness: int
    base_completeness: int
    fine_tuned_completeness: int
    base_safety: int
    fine_tuned_safety: int
    preference: ReviewPreference
    review_notes: str


@dataclass(frozen=True, slots=True)
class ReviewOutcome:
    """Artefatos produzidos por uma revisão humana consolidada."""

    output_path: Path
    completed_review_path: Path
    summary_path: Path
    decision_path: Path
    manifest_path: Path
    decision: ReviewDecision
    summary: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "output_path": str(self.output_path),
            "completed_review_path": str(self.completed_review_path),
            "summary_path": str(self.summary_path),
            "decision_path": str(self.decision_path),
            "manifest_path": str(self.manifest_path),
            "decision": self.decision,
            "summary": self.summary,
        }
