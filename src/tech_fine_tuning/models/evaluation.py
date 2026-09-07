"""Contratos neutros da comparação entre modelo base e adaptador."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

EvaluationSplit = Literal["validation", "test"]


@dataclass(frozen=True, slots=True)
class EvaluationExample:
    """Pergunta reservada, resposta de referência e mensagens do prompt."""

    record_id: str
    question: str
    reference_answer: str
    prompt_messages: tuple[dict[str, str], ...]


@dataclass(frozen=True, slots=True)
class EvaluationPlan:
    """Plano validado antes de carregar o modelo na GPU."""

    model_manifest_path: Path
    run_manifest_path: Path
    adapter_path: Path
    dataset_path: Path
    split_path: Path
    output_path: Path
    split: EvaluationSplit
    sample_size: int
    seed: int
    compare_base: bool
    model_id: str
    model_revision: str
    chat_template: str
    max_sequence_length: int
    load_in_4bit: bool
    max_new_tokens: int
    examples: tuple[EvaluationExample, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "paths": {
                "model_manifest": str(self.model_manifest_path),
                "run_manifest": str(self.run_manifest_path),
                "adapter": str(self.adapter_path),
                "dataset": str(self.dataset_path),
                "split": str(self.split_path),
                "output": str(self.output_path),
            },
            "evaluation": {
                "split": self.split,
                "sample_size": self.sample_size,
                "seed": self.seed,
                "compare_base": self.compare_base,
                "max_new_tokens": self.max_new_tokens,
            },
            "model": {
                "id": self.model_id,
                "revision": self.model_revision,
                "chat_template": self.chat_template,
                "max_sequence_length": self.max_sequence_length,
                "load_in_4bit": self.load_in_4bit,
            },
            "record_ids": [example.record_id for example in self.examples],
        }


@dataclass(frozen=True, slots=True)
class GeneratedComparison:
    """Texto e latência produzidos pelo backend para uma pergunta."""

    record_id: str
    fine_tuned_answer: str
    fine_tuned_latency_seconds: float
    base_answer: str | None
    base_latency_seconds: float | None


@dataclass(frozen=True, slots=True)
class EvaluationOutcome:
    """Arquivos públicos produzidos por uma comparação concluída."""

    output_path: Path
    responses_path: Path
    metrics_path: Path
    human_review_path: Path
    manifest_path: Path
    summary: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "output_path": str(self.output_path),
            "responses_path": str(self.responses_path),
            "metrics_path": str(self.metrics_path),
            "human_review_path": str(self.human_review_path),
            "manifest_path": str(self.manifest_path),
            "summary": self.summary,
        }
