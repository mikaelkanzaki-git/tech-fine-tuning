from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tech_fine_tuning.errors import DatasetReadError
from tech_fine_tuning.integrations.dataset.jsonl import (
    file_sha256,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


def test_json_and_jsonl_round_trip(tmp_path: Path) -> None:
    json_path = tmp_path / "nested" / "manifest.json"
    jsonl_path = tmp_path / "nested" / "records.jsonl"

    write_json(json_path, {"name": "MedQuAD", "count": 2})
    write_jsonl(jsonl_path, ({"id": 1}, {"id": 2}))

    assert read_json(json_path) == {"count": 2, "name": "MedQuAD"}
    assert read_jsonl(jsonl_path) == ({"id": 1}, {"id": 2})
    assert file_sha256(jsonl_path) == hashlib.sha256(jsonl_path.read_bytes()).hexdigest()


def test_read_json_rejects_non_object(tmp_path: Path) -> None:
    path = tmp_path / "list.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(DatasetReadError, match="objeto JSON"):
        read_json(path)


def test_read_jsonl_reports_invalid_line(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text('{"ok": true}\nnot-json\n', encoding="utf-8")

    with pytest.raises(DatasetReadError, match="linha 2"):
        read_jsonl(path)


def test_read_jsonl_rejects_non_object_and_ignores_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text("\n[1, 2]\n", encoding="utf-8")

    with pytest.raises(DatasetReadError, match="objeto JSON"):
        read_jsonl(path)


def test_readers_report_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DatasetReadError, match="Não foi possível ler"):
        read_json(tmp_path / "missing.json")
    with pytest.raises(DatasetReadError, match="Não foi possível ler"):
        read_jsonl(tmp_path / "missing.jsonl")
    with pytest.raises(DatasetReadError, match="calcular o hash"):
        file_sha256(tmp_path / "missing.jsonl")
