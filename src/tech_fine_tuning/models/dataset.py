"""Modelos do contrato canônico consumido e da projeção conversacional produzida."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

SplitName = Literal["train", "validation", "test"]
MessageRole = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class CanonicalTrainingRecord:
    """Projeção mínima do registro canônico necessária ao fine-tuning."""

    schema_version: str
    record_id: str
    document_id: str
    content_sha256: str
    question: str
    answer: str
    pii_status: str
    source_dataset: str
    source_relative_path: str
    source_upstream_revision: str | None
    source_url: str | None
    source_license: str


@dataclass(frozen=True, slots=True)
class SftMessage:
    """Mensagem independente do chat template de um modelo específico."""

    role: MessageRole
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True, slots=True)
class SftExample:
    """Exemplo conversacional e metadados excluídos do texto de treinamento."""

    messages: tuple[SftMessage, ...]
    canonical_record: CanonicalTrainingRecord

    def as_dict(self) -> dict[str, Any]:
        record = self.canonical_record
        return {
            "messages": [message.as_dict() for message in self.messages],
            "metadata": {
                "canonical_schema_version": record.schema_version,
                "record_id": record.record_id,
                "document_id": record.document_id,
                "content_sha256": record.content_sha256,
                "pii_status": record.pii_status,
                "source": {
                    "dataset": record.source_dataset,
                    "relative_path": record.source_relative_path,
                    "upstream_revision": record.source_upstream_revision,
                    "url": record.source_url,
                    "license": record.source_license,
                },
            },
        }


@dataclass(frozen=True, slots=True)
class PreparedSftDataset:
    """Resultado auditável da derivação do dataset para SFT."""

    manifest: dict[str, Any]
