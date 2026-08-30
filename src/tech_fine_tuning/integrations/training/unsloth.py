"""Backend Unsloth carregado somente no ambiente opcional de GPU."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from tech_fine_tuning.errors import TrainingDependencyError, TrainingExecutionError
from tech_fine_tuning.models.training import BackendTrainingResult, TrainingPlan


def _module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except (ImportError, OSError) as error:
        raise TrainingDependencyError(
            "O runtime de treinamento não está completo. "
            "Instale o extra 'training' ou use a imagem Docker do projeto."
        ) from error


def _limited(dataset: Any, limit: int | None) -> Any:
    if limit is None or limit >= len(dataset):
        return dataset
    return dataset.select(range(limit))


def _serializable_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[str(key)] = value
        elif hasattr(value, "item"):
            result[str(key)] = value.item()
        else:
            result[str(key)] = str(value)
    return result


def run_unsloth_training(plan: TrainingPlan) -> BackendTrainingResult:
    """Executa QLoRA sem expor Unsloth, TRL ou Transformers às demais camadas."""

    torch = _module("torch")
    datasets = _module("datasets")
    unsloth = _module("unsloth")
    chat_templates = _module("unsloth.chat_templates")
    trl = _module("trl")

    if not bool(torch.cuda.is_available()):
        raise TrainingDependencyError("O treinamento Unsloth requer uma GPU CUDA visível.")

    config = plan.config
    checkpoint_root = plan.output_path / "checkpoints"
    adapter_path = plan.output_path / "adapter"
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    try:
        model, tokenizer = unsloth.FastLanguageModel.from_pretrained(
            model_name=config.model.model_id,
            revision=config.model.model_revision,
            max_seq_length=config.model.max_sequence_length,
            load_in_4bit=config.model.load_in_4bit,
            load_in_8bit=False,
            full_finetuning=False,
        )
        model = unsloth.FastLanguageModel.get_peft_model(
            model,
            r=config.lora.rank,
            target_modules=list(config.lora.target_modules),
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            bias="none",
            use_gradient_checkpointing=config.lora.use_gradient_checkpointing,
            random_state=config.trainer.seed,
            use_rslora=config.lora.use_rslora,
            loftq_config=None,
        )
        tokenizer = chat_templates.get_chat_template(
            tokenizer,
            chat_template=config.model.chat_template,
        )

        train_dataset = datasets.load_dataset(
            "json", data_files=str(plan.train_path), split="train"
        )
        validation_dataset = datasets.load_dataset(
            "json", data_files=str(plan.validation_path), split="train"
        )
        train_dataset = _limited(train_dataset, config.dataset.train_limit)
        validation_dataset = _limited(
            validation_dataset, config.dataset.validation_limit
        )

        def format_messages(batch: Mapping[str, list[Any]]) -> dict[str, list[str]]:
            return {
                "text": [
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                    for messages in batch["messages"]
                ]
            }

        train_dataset = train_dataset.map(
            format_messages,
            batched=True,
            num_proc=config.dataset.num_proc,
        )
        validation_dataset = validation_dataset.map(
            format_messages,
            batched=True,
            num_proc=config.dataset.num_proc,
        )

        trainer_arguments: dict[str, Any] = {
            "output_dir": str(checkpoint_root),
            "per_device_train_batch_size": config.trainer.per_device_train_batch_size,
            "gradient_accumulation_steps": config.trainer.gradient_accumulation_steps,
            "learning_rate": config.trainer.learning_rate,
            "warmup_steps": config.trainer.warmup_steps,
            "logging_steps": config.trainer.logging_steps,
            "save_strategy": "steps",
            "save_steps": config.trainer.save_steps,
            "eval_strategy": "steps",
            "eval_steps": config.trainer.eval_steps,
            "save_total_limit": config.trainer.save_total_limit,
            "optim": config.trainer.optimizer,
            "weight_decay": config.trainer.weight_decay,
            "lr_scheduler_type": config.trainer.lr_scheduler_type,
            "seed": config.trainer.seed,
            "report_to": config.trainer.report_to,
            "bf16": bool(torch.cuda.is_bf16_supported()),
            "fp16": not bool(torch.cuda.is_bf16_supported()),
            "dataset_text_field": "text",
            "dataset_num_proc": config.dataset.num_proc,
            "max_length": config.model.max_sequence_length,
        }
        if config.trainer.max_steps is not None:
            trainer_arguments["max_steps"] = config.trainer.max_steps
        else:
            trainer_arguments["num_train_epochs"] = config.trainer.num_train_epochs

        trainer = trl.SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            args=trl.SFTConfig(**trainer_arguments),
        )
        if config.dataset.train_on_responses_only:
            trainer = chat_templates.train_on_responses_only(
                trainer,
                instruction_part="<|im_start|>user\n",
                response_part="<|im_start|>assistant\n",
            )

        train_result = trainer.train(
            resume_from_checkpoint=(
                str(plan.resume_from_checkpoint) if plan.resume_from_checkpoint else None
            )
        )
        metrics = {
            "train": _serializable_metrics(train_result.metrics),
            "validation": _serializable_metrics(trainer.evaluate()),
        }
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
        peak_memory = float(torch.cuda.max_memory_reserved()) / (1024**3)
    except TrainingDependencyError:
        raise
    except Exception as error:
        raise TrainingExecutionError(
            f"O backend Unsloth interrompeu o treinamento: {type(error).__name__}."
        ) from error

    return BackendTrainingResult(
        metrics=metrics,
        adapter_path=adapter_path,
        peak_reserved_memory_gib=round(peak_memory, 3),
    )
