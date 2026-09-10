from __future__ import annotations

import json
from pathlib import Path

import pytest

from tech_fine_tuning import runner
from tech_fine_tuning.models.audit import SftAuditOutcome
from tech_fine_tuning.models.curation import SftCurationOutcome
from tech_fine_tuning.models.evaluation import EvaluationOutcome
from tech_fine_tuning.models.review import ReviewOutcome
from tech_fine_tuning.runner import main

from .test_sft_dataset_service import create_source


def test_prepare_sft_command_writes_artifacts(tmp_path: Path) -> None:
    source = create_source(tmp_path)
    output = tmp_path / "sft"

    exit_code = main(
        [
            "prepare-sft",
            "--source",
            str(source),
            "--output",
            str(output),
            "--system-prompt",
            "Safe system",
        ]
    )

    assert exit_code == 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["summary"]["examples"] == 4
    assert (output / "train.jsonl").exists()
    assert (output / "validation.jsonl").exists()
    assert (output / "test.jsonl").exists()


def test_prepare_sft_command_returns_error_for_invalid_source(tmp_path: Path) -> None:
    exit_code = main(["prepare-sft", "--source", str(tmp_path / "missing")])

    assert exit_code == 2


def test_evaluate_command_forwards_comparison_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_execute(**kwargs: object) -> EvaluationOutcome:
        captured.update(kwargs)
        output = tmp_path / "evaluation"
        return EvaluationOutcome(
            output_path=output,
            responses_path=output / "responses.jsonl",
            metrics_path=output / "automatic-metrics.json",
            human_review_path=output / "human-review.csv",
            manifest_path=output / "evaluation-manifest.json",
            summary={"examples": 20},
        )

    monkeypatch.setattr(runner, "execute_evaluation", fake_execute)
    exit_code = main(
        [
            "evaluate",
            "--manifest",
            str(tmp_path / "model-manifest.json"),
            "--source",
            str(tmp_path / "sft"),
            "--output",
            str(tmp_path / "evaluation"),
            "--split",
            "validation",
            "--sample-size",
            "20",
            "--temperature",
            "0.6",
            "--top-p",
            "0.75",
            "--top-k",
            "16",
            "--repetition-penalty",
            "1.15",
            "--compare-base",
        ]
    )

    assert exit_code == 0
    assert captured["sample_size"] == 20
    assert captured["split"] == "validation"
    assert captured["compare_base"] is True
    assert captured["do_sample"] is True
    assert captured["temperature"] == 0.6
    assert captured["top_p"] == 0.75
    assert captured["top_k"] == 16
    assert captured["repetition_penalty"] == 1.15


def test_summarize_review_command_forwards_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_execute(**kwargs: object) -> ReviewOutcome:
        captured.update(kwargs)
        output = tmp_path / "review-result"
        return ReviewOutcome(
            output_path=output,
            completed_review_path=output / "human-review-completed.csv",
            summary_path=output / "human-review-summary.json",
            decision_path=output / "decision.md",
            manifest_path=output / "review-manifest.json",
            decision="rejected",
            summary={"records": 20},
        )

    monkeypatch.setattr(runner, "execute_review_summary", fake_execute)
    review = tmp_path / "human-review.csv"
    manifest = tmp_path / "evaluation-manifest.json"
    output = tmp_path / "review-result"
    exit_code = main(
        [
            "summarize-review",
            "--review",
            str(review),
            "--evaluation-manifest",
            str(manifest),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert captured == {
        "review_path": review,
        "evaluation_manifest_path": manifest,
        "output_path": output,
    }


def test_audit_sft_command_forwards_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_execute(**kwargs: object) -> SftAuditOutcome:
        captured.update(kwargs)
        output = tmp_path / "audit"
        return SftAuditOutcome(
            output_path=output,
            summary_path=output / "audit-summary.json",
            issues_path=output / "issues.jsonl",
            report_path=output / "report.md",
            manifest_path=output / "audit-manifest.json",
            summary={"examples": 4},
        )

    monkeypatch.setattr(runner, "audit_sft_dataset", fake_execute)
    source = tmp_path / "sft"
    output = tmp_path / "audit"
    exit_code = main(
        [
            "audit-sft",
            "--source",
            str(source),
            "--output",
            str(output),
            "--max-sequence-length",
            "1024",
        ]
    )

    assert exit_code == 0
    assert captured == {
        "source_path": source,
        "output_path": output,
        "max_sequence_length": 1024,
    }


def test_curate_sft_command_forwards_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_execute(**kwargs: object) -> SftCurationOutcome:
        captured.update(kwargs)
        output = tmp_path / "curated"
        return SftCurationOutcome(
            output_path=output,
            manifest_path=output / "manifest.json",
            excluded_path=output / "excluded.jsonl",
            summary={"output_examples": 3},
        )

    monkeypatch.setattr(runner, "curate_sft_dataset", fake_execute)
    source = tmp_path / "sft"
    output = tmp_path / "curated"
    exit_code = main(
        ["curate-sft", "--source", str(source), "--output", str(output)]
    )

    assert exit_code == 0
    assert captured == {
        "source_path": source,
        "output_path": output,
        "system_prompt": None,
    }
