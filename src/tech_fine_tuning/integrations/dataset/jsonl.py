"""Persistência dos artefatos JSON e JSONL de treinamento."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from tech_fine_tuning.errors import DatasetReadError


def read_json(path: Path) -> dict[str, Any]:
    """Lê um objeto JSON e traduz falhas para o erro estável da aplicação."""

    try:
        with path.open(encoding="utf-8") as stream:
            data = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DatasetReadError(f"Não foi possível ler {path}: {error}") from error
    if not isinstance(data, dict):
        raise DatasetReadError(f"O arquivo {path} deve conter um objeto JSON.")
    return data


def read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    """Lê objetos JSONL preservando o número da linha em mensagens de erro."""

    records: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as error:
                    raise DatasetReadError(
                        f"JSON inválido em {path}, linha {line_number}: {error}"
                    ) from error
                if not isinstance(data, dict):
                    raise DatasetReadError(
                        f"{path}, linha {line_number}, deve conter um objeto JSON."
                    )
                records.append(data)
    except (OSError, UnicodeError) as error:
        raise DatasetReadError(f"Não foi possível ler {path}: {error}") from error
    return tuple(records)


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    """Grava um manifesto JSON legível e determinístico."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    """Grava um objeto JSON por linha."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            stream.write("\n")


def file_sha256(path: Path) -> str:
    """Calcula a identidade do artefato sem carregar o arquivo inteiro em memória."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise DatasetReadError(f"Não foi possível calcular o hash de {path}: {error}") from error
    return digest.hexdigest()
