from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tech_fine_tuning.errors import ReviewValidationError
from tech_fine_tuning.integrations.dataset.jsonl import file_sha256, write_json
from tech_fine_tuning.services.review_service import execute_review_summary

FIELDNAMES = [
    "record_id",
    "question",
    "reference_answer",
    "base_answer",
    "fine_tuned_answer",
    "base_correctness_0_2",
    "fine_tuned_correctness_0_2",
    "base_completeness_0_2",
    "fine_tuned_completeness_0_2",
    "base_safety_0_2",
    "fine_tuned_safety_0_2",
    "preference_base_fine_tuned_tie",
    "review_notes",
]


def _row(record_id: str, *, preference: str = "base") -> dict[str, str]:
    return {
        "record_id": record_id,
        "question": "Question",
        "reference_answer": "Reference",
        "base_answer": "Base",
        "fine_tuned_answer": "Fine-tuned",
        "base_correctness_0_2": "2",
        "fine_tuned_correctness_0_2": "0" if preference == "base" else "2",
        "base_completeness_0_2": "2",
        "fine_tuned_completeness_0_2": "1" if preference == "base" else "2",
        "base_safety_0_2": "2",
        "fine_tuned_safety_0_2": "0" if preference == "base" else "2",
        "preference_base_fine_tuned_tie": preference,
        "review_notes": "Erro factual" if preference == "base" else "",
    }


def _write_review(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _write_manifest(path: Path, rows: list[dict[str, str]], review_path: Path) -> None:
    write_json(
        path,
        {
            "schema_version": "1.0",
            "status": "completed",
            "plan": {
                "evaluation": {"compare_base": True, "split": "validation"},
                "record_ids": [row["record_id"] for row in rows],
            },
            "outputs_sha256": {"human_review": file_sha256(review_path)},
            "summary": {
                "examples": len(rows),
                "base_empty_answers": 0,
                "fine_tuned_empty_answers": 0,
                "token_f1_base_wins": 0,
                "token_f1_tuned_wins": len(rows),
                "token_f1_ties": 0,
            },
        },
    )


def test_review_summary_rejects_regression_and_preserves_completed_csv(tmp_path: Path) -> None:
    review_path = tmp_path / "human-review.csv"
    rows = [_row("record-1"), _row("record-2", preference="tie")]
    _write_review(review_path, rows)
    manifest_path = tmp_path / "evaluation-manifest.json"
    _write_manifest(manifest_path, rows, review_path)
    original = review_path.read_bytes()
    rows[1]["preference_base_fine_tuned_tie"] = "base"
    rows[1]["fine_tuned_correctness_0_2"] = "0"
    rows[1]["fine_tuned_completeness_0_2"] = "1"
    rows[1]["fine_tuned_safety_0_2"] = "0"
    rows[1]["review_notes"] = "Outra regressão"
    _write_review(review_path, rows)

    outcome = execute_review_summary(
        review_path=review_path,
        evaluation_manifest_path=manifest_path,
        output_path=tmp_path / "review-result",
    )

    assert outcome.decision == "rejected"
    assert outcome.completed_review_path.read_bytes() == review_path.read_bytes()
    assert outcome.completed_review_path.read_bytes() != original
    assert outcome.summary["preferences"] == {"base": 2, "fine_tuned": 0, "tie": 0}
    assert outcome.summary["safety"]["severe_regressions"] == 2
    assert outcome.summary["quality_gate"]["passed"] is False
    assert outcome.summary["provenance"]["changed_since_generation"] is True
    assert "**Status: REPROVADO**" in outcome.decision_path.read_text(encoding="utf-8")
    result_manifest = json.loads(outcome.manifest_path.read_text(encoding="utf-8"))
    assert result_manifest["status"] == "rejected"
    assert result_manifest["outputs_sha256"]["human_review_completed"] == file_sha256(
        outcome.completed_review_path
    )


def test_review_summary_approves_when_all_gates_pass(tmp_path: Path) -> None:
    review_path = tmp_path / "human-review.csv"
    rows = [_row("record-1", preference="fine_tuned"), _row("record-2", preference="tie")]
    _write_review(review_path, rows)
    manifest_path = tmp_path / "evaluation-manifest.json"
    _write_manifest(manifest_path, rows, review_path)

    outcome = execute_review_summary(
        review_path=review_path,
        evaluation_manifest_path=manifest_path,
        output_path=tmp_path / "review-result",
    )

    assert outcome.decision == "approved"
    assert outcome.summary["quality_gate"]["passed"] is True
    assert outcome.summary["preferences"]["fine_tuned"] == 1


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("base_correctness_0_2", "", "deve ser 0, 1 ou 2"),
        ("fine_tuned_safety_0_2", "3", "deve ser 0, 1 ou 2"),
        ("preference_base_fine_tuned_tie", "tuned", "deve ser base"),
    ],
)
def test_review_summary_rejects_invalid_answers(
    tmp_path: Path, column: str, value: str, message: str
) -> None:
    review_path = tmp_path / "human-review.csv"
    rows = [_row("record-1")]
    rows[0][column] = value
    _write_review(review_path, rows)
    manifest_path = tmp_path / "evaluation-manifest.json"
    _write_manifest(manifest_path, rows, review_path)

    with pytest.raises(ReviewValidationError, match=message):
        execute_review_summary(
            review_path=review_path,
            evaluation_manifest_path=manifest_path,
            output_path=tmp_path / "review-result",
        )


def test_review_summary_rejects_record_ids_outside_manifest(tmp_path: Path) -> None:
    review_path = tmp_path / "human-review.csv"
    rows = [_row("record-1")]
    _write_review(review_path, rows)
    manifest_path = tmp_path / "evaluation-manifest.json"
    _write_manifest(manifest_path, rows, review_path)
    rows[0]["record_id"] = "other-record"
    _write_review(review_path, rows)

    with pytest.raises(ReviewValidationError, match="não correspondem"):
        execute_review_summary(
            review_path=review_path,
            evaluation_manifest_path=manifest_path,
            output_path=tmp_path / "review-result",
        )


def test_review_summary_refuses_nonempty_output(tmp_path: Path) -> None:
    review_path = tmp_path / "human-review.csv"
    rows = [_row("record-1")]
    _write_review(review_path, rows)
    manifest_path = tmp_path / "evaluation-manifest.json"
    _write_manifest(manifest_path, rows, review_path)
    output_path = tmp_path / "review-result"
    output_path.mkdir()
    (output_path / "keep.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(ReviewValidationError, match="não está vazia"):
        execute_review_summary(
            review_path=review_path,
            evaluation_manifest_path=manifest_path,
            output_path=output_path,
        )
