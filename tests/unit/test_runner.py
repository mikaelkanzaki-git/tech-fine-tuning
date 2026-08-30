from __future__ import annotations

import json
from pathlib import Path

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
