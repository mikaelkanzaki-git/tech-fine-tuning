from __future__ import annotations

from typing import Any

from tech_fine_tuning.integrations.dataset.sampling import (
    balanced_source_indices,
    selection_indices,
    source_distribution,
    source_group,
)


def _record(record_id: str, source: str) -> dict[str, Any]:
    return {
        "metadata": {
            "record_id": record_id,
            "source": {
                "dataset": "MedQuAD",
                "relative_path": f"{source}/{record_id}.xml",
            },
        }
    }


def test_balanced_selection_is_reproducible_and_source_balanced() -> None:
    records = [
        _record(f"{source}-{index}", source)
        for source in ("source-a", "source-b", "source-c")
        for index in range(8)
    ]

    indices = balanced_source_indices(records, limit=8, seed=3407)
    selected_ids = {records[index]["metadata"]["record_id"] for index in indices}
    reversed_records = list(reversed(records))
    reversed_indices = balanced_source_indices(reversed_records, limit=8, seed=3407)
    reversed_ids = {
        reversed_records[index]["metadata"]["record_id"] for index in reversed_indices
    }

    assert selected_ids == reversed_ids
    assert source_distribution(records, indices) == {
        "source-a": 3,
        "source-b": 3,
        "source-c": 2,
    }


def test_balanced_selection_redistributes_capacity_from_a_small_source() -> None:
    records = [_record("a-0", "source-a")]
    records.extend(_record(f"b-{index}", "source-b") for index in range(10))
    records.extend(_record(f"c-{index}", "source-c") for index in range(10))

    indices = balanced_source_indices(records, limit=9, seed=7)

    assert source_distribution(records, indices) == {
        "source-a": 1,
        "source-b": 4,
        "source-c": 4,
    }


def test_selection_preserves_legacy_head_strategy_and_skips_unneeded_cut() -> None:
    records = [_record(str(index), "source") for index in range(4)]

    assert selection_indices(records, limit=2, strategy="head", seed=1) == (0, 1)
    assert selection_indices(records, limit=None, strategy="balanced_by_source", seed=1) is None
    assert selection_indices(records, limit=4, strategy="balanced_by_source", seed=1) is None


def test_source_group_has_safe_fallbacks() -> None:
    assert source_group(_record("1", "collection")) == "collection"
    assert source_group({"metadata": {"source": {"dataset": "MedQuAD"}}}) == "MedQuAD"
    assert source_group({}) == "unknown"
