"""Contratos internos de configuração e execução do fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelTrainingConfig:
    """Modelo e tokenizer imutáveis usados como ponto de partida."""

    model_id: str
    model_revision: str
    base_model_id: str
    base_model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    chat_template: str
    max_sequence_length: int
    load_in_4bit: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.model_id,
            "revision": self.model_revision,
            "base_id": self.base_model_id,
            "base_revision": self.base_model_revision,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_revision": self.tokenizer_revision,
            "chat_template": self.chat_template,
            "max_sequence_length": self.max_sequence_length,
            "load_in_4bit": self.load_in_4bit,
        }


@dataclass(frozen=True, slots=True)
class LoraTrainingConfig:
    """Parâmetros do adaptador LoRA/QLoRA."""

    rank: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]
    use_gradient_checkpointing: str
    use_rslora: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": list(self.target_modules),
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_rslora": self.use_rslora,
        }


@dataclass(frozen=True, slots=True)
class DatasetTrainingConfig:
    """Limites de dados aplicados sem misturar os splits."""

    train_limit: int | None
    validation_limit: int | None
    num_proc: int
    train_on_responses_only: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "train_limit": self.train_limit,
            "validation_limit": self.validation_limit,
            "num_proc": self.num_proc,
            "train_on_responses_only": self.train_on_responses_only,
        }


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    """Hiperparâmetros independentes da infraestrutura de execução."""

    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    max_steps: int | None
    num_train_epochs: float | None
    learning_rate: float
    warmup_steps: int
    logging_steps: int
    save_steps: int
    eval_steps: int
    save_total_limit: int
    optimizer: str
    weight_decay: float
    lr_scheduler_type: str
    seed: int
    report_to: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_steps": self.max_steps,
            "num_train_epochs": self.num_train_epochs,
            "learning_rate": self.learning_rate,
            "warmup_steps": self.warmup_steps,
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
            "eval_steps": self.eval_steps,
            "save_total_limit": self.save_total_limit,
            "optimizer": self.optimizer,
            "weight_decay": self.weight_decay,
            "lr_scheduler_type": self.lr_scheduler_type,
            "seed": self.seed,
            "report_to": self.report_to,
        }


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Configuração completa carregada de um perfil versionado."""

    schema_version: str
    run_name: str
    model: ModelTrainingConfig
    lora: LoraTrainingConfig
    dataset: DatasetTrainingConfig
    trainer: TrainerConfig

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_name": self.run_name,
            "model": self.model.as_dict(),
            "lora": self.lora.as_dict(),
            "dataset": self.dataset.as_dict(),
            "trainer": self.trainer.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    """Plano validado que pode ser executado em qualquer ambiente."""

    config: TrainingConfig
    config_path: Path
    source_path: Path
    output_path: Path
    train_path: Path
    validation_path: Path
    dataset_manifest_sha256: str
    train_examples: int
    validation_examples: int
    producer_commit: str
    resume_from_checkpoint: Path | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.as_dict(),
            "paths": {
                "config": str(self.config_path),
                "source": str(self.source_path),
                "output": str(self.output_path),
                "train": str(self.train_path),
                "validation": str(self.validation_path),
                "resume_from_checkpoint": (
                    str(self.resume_from_checkpoint) if self.resume_from_checkpoint else None
                ),
            },
            "dataset": {
                "manifest_sha256": self.dataset_manifest_sha256,
                "train_examples": self.train_examples,
                "validation_examples": self.validation_examples,
            },
            "producer_commit": self.producer_commit,
        }


@dataclass(frozen=True, slots=True)
class BackendTrainingResult:
    """Resultado neutro retornado pela integração de treinamento."""

    metrics: dict[str, Any]
    adapter_path: Path
    peak_reserved_memory_gib: float | None


@dataclass(frozen=True, slots=True)
class TrainingOutcome:
    """Resultado público da orquestração de uma run."""

    plan: TrainingPlan
    dry_run: bool
    run_manifest_path: Path | None = None
    model_manifest_path: Path | None = None

    def as_dict(self) -> dict[str, Any]:
        data = self.plan.as_dict()
        data["dry_run"] = self.dry_run
        data["run_manifest_path"] = (
            str(self.run_manifest_path) if self.run_manifest_path else None
        )
        data["model_manifest_path"] = (
            str(self.model_manifest_path) if self.model_manifest_path else None
        )
        return data
