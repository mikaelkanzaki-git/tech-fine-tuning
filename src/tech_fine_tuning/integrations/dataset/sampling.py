"""Seleção reproduzível de exemplos antes de entregá-los ao backend de treino."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from tech_fine_tuning.models.training import DatasetSamplingStrategy


def _metadata(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = record.get("metadata")
    return value if isinstance(value, dict) else {}


def source_group(record: Mapping[str, Any]) -> str:
    """Identifica a coleção MedQuAD pela primeira parte do caminho de origem."""

    metadata = _metadata(record)
    source = metadata.get("source")
    if not isinstance(source, dict):
        return "unknown"
    relative_path = source.get("relative_path")
    if isinstance(relative_path, str) and relative_path.strip():
        return relative_path.replace("\\", "/").split("/", maxsplit=1)[0]
    dataset = source.get("dataset")
    if isinstance(dataset, str) and dataset.strip():
        return dataset.strip()
    return "unknown"


def _record_id(record: Mapping[str, Any], index: int) -> str:
    value = _metadata(record).get("record_id")
    return value if isinstance(value, str) and value else f"index:{index}"


def _selection_key(*, seed: int, namespace: str, record_id: str) -> bytes:
    return hashlib.sha256(f"{seed}:{namespace}:{record_id}".encode()).digest()


def balanced_source_indices(
    records: Sequence[Mapping[str, Any]],
    *,
    limit: int,
    seed: int,
) -> tuple[int, ...]:
    """Distribui o limite em rodadas entre fontes e embaralha por hash estável."""

    if limit <= 0:
        raise ValueError("limit deve ser positivo")
    if limit >= len(records):
        return tuple(range(len(records)))

    grouped: dict[str, list[tuple[bytes, str, int]]] = defaultdict(list)
    for index in range(len(records)):
        record = records[index]
        record_id = _record_id(record, index)
        group = source_group(record)
        grouped[group].append(
            (
                _selection_key(seed=seed, namespace=f"source:{group}", record_id=record_id),
                record_id,
                index,
            )
        )
    for candidates in grouped.values():
        candidates.sort()

    selected: list[tuple[str, int]] = []
    positions = {group: 0 for group in grouped}
    groups = sorted(grouped)
    while len(selected) < limit:
        progressed = False
        for group in groups:
            position = positions[group]
            candidates = grouped[group]
            if position >= len(candidates):
                continue
            _, record_id, index = candidates[position]
            selected.append((record_id, index))
            positions[group] += 1
            progressed = True
            if len(selected) == limit:
                break
        if not progressed:
            break

    selected.sort(
        key=lambda item: (
            _selection_key(seed=seed, namespace="final", record_id=item[0]),
            item[0],
            item[1],
        )
    )
    return tuple(index for _, index in selected)


def selection_indices(
    records: Sequence[Mapping[str, Any]],
    *,
    limit: int | None,
    strategy: DatasetSamplingStrategy,
    seed: int,
) -> tuple[int, ...] | None:
    """Retorna índices selecionados ou ``None`` quando não há corte a aplicar."""

    if limit is None or limit >= len(records):
        return None
    if strategy == "head":
        return tuple(range(limit))
    return balanced_source_indices(records, limit=limit, seed=seed)


def source_distribution(
    records: Sequence[Mapping[str, Any]],
    indices: Sequence[int] | None = None,
) -> dict[str, int]:
    """Conta exemplos por coleção para registrar a amostra efetiva."""

    selected = indices if indices is not None else range(len(records))
    counts = Counter(source_group(records[index]) for index in selected)
    return dict(sorted(counts.items()))
