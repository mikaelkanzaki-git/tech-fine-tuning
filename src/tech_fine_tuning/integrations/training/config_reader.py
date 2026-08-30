"""Leitura estrita dos perfis TOML de treinamento."""

from __future__ import annotations

import re
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tech_fine_tuning.errors import TrainingConfigurationError
from tech_fine_tuning.models.training import (
    DatasetTrainingConfig,
    LoraTrainingConfig,
    ModelTrainingConfig,
    TrainerConfig,
    TrainingConfig,
)

TRAINING_CONFIG_SCHEMA_VERSION = "1.0"
_REVISION = re.compile(r"^[0-9a-f]{7,64}$")


def _mapping(data: Mapping[str, Any], field: str, *, context: str) -> Mapping[str, Any]:
    value = data.get(field)
    if not isinstance(value, dict):
        raise TrainingConfigurationError(f"{context}: seção {field!r} é obrigatória.")
    return value


def _reject_unknown(data: Mapping[str, Any], allowed: set[str], *, context: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise TrainingConfigurationError(
            f"{context}: campos desconhecidos: {', '.join(unknown)}."
        )


def _string(data: Mapping[str, Any], field: str, *, context: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise TrainingConfigurationError(
            f"{context}: campo {field!r} deve ser uma string não vazia."
        )
    return value.strip()


def _integer(
    data: Mapping[str, Any],
    field: str,
    *,
    context: str,
    default: int | None = None,
) -> int:
    value = data.get(field, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TrainingConfigurationError(f"{context}: campo {field!r} deve ser inteiro.")
    return value


def _optional_positive_integer(
    data: Mapping[str, Any], field: str, *, context: str
) -> int | None:
    if field not in data:
        return None
    value = _integer(data, field, context=context)
    if value <= 0:
        raise TrainingConfigurationError(f"{context}: campo {field!r} deve ser positivo.")
    return value


def _number(
    data: Mapping[str, Any], field: str, *, context: str, default: float | None = None
) -> float:
    value = data.get(field, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TrainingConfigurationError(f"{context}: campo {field!r} deve ser numérico.")
    return float(value)


def _boolean(
    data: Mapping[str, Any], field: str, *, context: str, default: bool | None = None
) -> bool:
    value = data.get(field, default)
    if not isinstance(value, bool):
        raise TrainingConfigurationError(f"{context}: campo {field!r} deve ser booleano.")
    return value


def _strings(data: Mapping[str, Any], field: str, *, context: str) -> tuple[str, ...]:
    value = data.get(field)
    if not isinstance(value, list) or not value:
        raise TrainingConfigurationError(f"{context}: campo {field!r} deve ser uma lista.")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise TrainingConfigurationError(
            f"{context}: campo {field!r} aceita somente strings não vazias."
        )
    return tuple(item.strip() for item in value)


def _validate_revision(value: str, *, context: str) -> str:
    if not _REVISION.fullmatch(value):
        raise TrainingConfigurationError(
            f"{context}: use uma revisão Git imutável com pelo menos 7 caracteres hexadecimais."
        )
    return value


def _read_toml(path: Path) -> Mapping[str, Any]:
    try:
        with path.expanduser().resolve().open("rb") as stream:
            data = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise TrainingConfigurationError(
            f"Não foi possível ler a configuração {path}: {error}"
        ) from error
    return data


def read_training_config(path: Path) -> TrainingConfig:
    """Lê um perfil sem aceitar opções silenciosamente ignoradas."""

    data = _read_toml(path)
    context = f"configuração {path}"
    _reject_unknown(
        data,
        {"schema_version", "run_name", "model", "lora", "dataset", "trainer"},
        context=context,
    )
    schema_version = _string(data, "schema_version", context=context)
    if schema_version != TRAINING_CONFIG_SCHEMA_VERSION:
        raise TrainingConfigurationError(
            f"{context}: schema {schema_version!r} não suportado; "
            f"esperado {TRAINING_CONFIG_SCHEMA_VERSION!r}."
        )

    model_data = _mapping(data, "model", context=context)
    model_context = f"{context}, seção model"
    _reject_unknown(
        model_data,
        {
            "id",
            "revision",
            "base_id",
            "base_revision",
            "tokenizer_id",
            "tokenizer_revision",
            "chat_template",
            "max_sequence_length",
            "load_in_4bit",
        },
        context=model_context,
    )
    max_sequence_length = _integer(
        model_data, "max_sequence_length", context=model_context
    )
    if max_sequence_length <= 0:
        raise TrainingConfigurationError(
            f"{model_context}: max_sequence_length deve ser positivo."
        )
    model = ModelTrainingConfig(
        model_id=_string(model_data, "id", context=model_context),
        model_revision=_validate_revision(
            _string(model_data, "revision", context=model_context), context=model_context
        ),
        base_model_id=_string(model_data, "base_id", context=model_context),
        base_model_revision=_validate_revision(
            _string(model_data, "base_revision", context=model_context), context=model_context
        ),
        tokenizer_id=_string(model_data, "tokenizer_id", context=model_context),
        tokenizer_revision=_validate_revision(
            _string(model_data, "tokenizer_revision", context=model_context),
            context=model_context,
        ),
        chat_template=_string(model_data, "chat_template", context=model_context),
        max_sequence_length=max_sequence_length,
        load_in_4bit=_boolean(model_data, "load_in_4bit", context=model_context),
    )

    lora_data = _mapping(data, "lora", context=context)
    lora_context = f"{context}, seção lora"
    _reject_unknown(
        lora_data,
        {
            "rank",
            "alpha",
            "dropout",
            "target_modules",
            "use_gradient_checkpointing",
            "use_rslora",
        },
        context=lora_context,
    )
    rank = _integer(lora_data, "rank", context=lora_context)
    alpha = _integer(lora_data, "alpha", context=lora_context)
    dropout = _number(lora_data, "dropout", context=lora_context)
    if rank <= 0 or alpha <= 0 or not 0 <= dropout < 1:
        raise TrainingConfigurationError(
            f"{lora_context}: rank/alpha devem ser positivos e dropout deve estar em [0, 1)."
        )
    lora = LoraTrainingConfig(
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_modules=_strings(lora_data, "target_modules", context=lora_context),
        use_gradient_checkpointing=_string(
            lora_data, "use_gradient_checkpointing", context=lora_context
        ),
        use_rslora=_boolean(lora_data, "use_rslora", context=lora_context),
    )

    dataset_data = _mapping(data, "dataset", context=context)
    dataset_context = f"{context}, seção dataset"
    _reject_unknown(
        dataset_data,
        {"train_limit", "validation_limit", "num_proc", "train_on_responses_only"},
        context=dataset_context,
    )
    num_proc = _integer(dataset_data, "num_proc", context=dataset_context, default=1)
    if num_proc <= 0:
        raise TrainingConfigurationError(f"{dataset_context}: num_proc deve ser positivo.")
    dataset = DatasetTrainingConfig(
        train_limit=_optional_positive_integer(
            dataset_data, "train_limit", context=dataset_context
        ),
        validation_limit=_optional_positive_integer(
            dataset_data, "validation_limit", context=dataset_context
        ),
        num_proc=num_proc,
        train_on_responses_only=_boolean(
            dataset_data,
            "train_on_responses_only",
            context=dataset_context,
            default=True,
        ),
    )

    trainer_data = _mapping(data, "trainer", context=context)
    trainer_context = f"{context}, seção trainer"
    _reject_unknown(
        trainer_data,
        {
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "max_steps",
            "num_train_epochs",
            "learning_rate",
            "warmup_steps",
            "logging_steps",
            "save_steps",
            "eval_steps",
            "save_total_limit",
            "optimizer",
            "weight_decay",
            "lr_scheduler_type",
            "seed",
            "report_to",
        },
        context=trainer_context,
    )
    max_steps = _optional_positive_integer(trainer_data, "max_steps", context=trainer_context)
    num_train_epochs = (
        _number(trainer_data, "num_train_epochs", context=trainer_context)
        if "num_train_epochs" in trainer_data
        else None
    )
    if (max_steps is None) == (num_train_epochs is None):
        raise TrainingConfigurationError(
            f"{trainer_context}: informe exatamente um de max_steps ou num_train_epochs."
        )
    if num_train_epochs is not None and num_train_epochs <= 0:
        raise TrainingConfigurationError(
            f"{trainer_context}: num_train_epochs deve ser positivo."
        )
    trainer = TrainerConfig(
        per_device_train_batch_size=_integer(
            trainer_data, "per_device_train_batch_size", context=trainer_context
        ),
        gradient_accumulation_steps=_integer(
            trainer_data, "gradient_accumulation_steps", context=trainer_context
        ),
        max_steps=max_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=_number(trainer_data, "learning_rate", context=trainer_context),
        warmup_steps=_integer(trainer_data, "warmup_steps", context=trainer_context, default=0),
        logging_steps=_integer(trainer_data, "logging_steps", context=trainer_context),
        save_steps=_integer(trainer_data, "save_steps", context=trainer_context),
        eval_steps=_integer(trainer_data, "eval_steps", context=trainer_context),
        save_total_limit=_integer(trainer_data, "save_total_limit", context=trainer_context),
        optimizer=_string(trainer_data, "optimizer", context=trainer_context),
        weight_decay=_number(trainer_data, "weight_decay", context=trainer_context),
        lr_scheduler_type=_string(
            trainer_data, "lr_scheduler_type", context=trainer_context
        ),
        seed=_integer(trainer_data, "seed", context=trainer_context),
        report_to=_string(trainer_data, "report_to", context=trainer_context),
    )
    positive_integers = {
        "per_device_train_batch_size": trainer.per_device_train_batch_size,
        "gradient_accumulation_steps": trainer.gradient_accumulation_steps,
        "logging_steps": trainer.logging_steps,
        "save_steps": trainer.save_steps,
        "eval_steps": trainer.eval_steps,
        "save_total_limit": trainer.save_total_limit,
    }
    invalid = [name for name, value in positive_integers.items() if value <= 0]
    invalid_numeric_value = (
        trainer.learning_rate <= 0
        or trainer.weight_decay < 0
        or trainer.warmup_steps < 0
    )
    if invalid or invalid_numeric_value:
        raise TrainingConfigurationError(
            f"{trainer_context}: hiperparâmetros numéricos fora dos limites aceitos."
        )

    return TrainingConfig(
        schema_version=schema_version,
        run_name=_string(data, "run_name", context=context),
        model=model,
        lora=lora,
        dataset=dataset,
        trainer=trainer,
    )
