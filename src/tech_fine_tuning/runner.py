"""Interface de linha de comando do serviço de fine-tuning."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from tech_fine_tuning.config.settings import Settings
from tech_fine_tuning.errors import (
    ConfigurationError,
    DatasetReadError,
    SftPreparationError,
    TrainingConfigurationError,
    TrainingDependencyError,
    TrainingExecutionError,
    TrainingPreflightError,
)
from tech_fine_tuning.integrations.system.diagnostics import collect_environment_diagnostic
from tech_fine_tuning.services.sft_dataset_service import prepare_sft_dataset
from tech_fine_tuning.services.training_service import execute_training


def _build_parser(settings: Settings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tech-fine-tuning",
        description="Prepara, treina e avalia o modelo médico do Tech Challenge.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subcommands.add_parser(
        "prepare-sft",
        help="Converte os splits canônicos em exemplos conversacionais para SFT.",
    )
    prepare_parser.add_argument(
        "--source",
        type=Path,
        default=settings.canonical_dataset_path,
        help="Diretório gerado pelo tech-ingestao.",
    )
    prepare_parser.add_argument(
        "--output",
        type=Path,
        default=settings.sft_output_path,
        help="Diretório dos JSONL conversacionais.",
    )
    prepare_parser.add_argument(
        "--system-prompt",
        default=settings.system_prompt,
        help="Instrução de sistema repetida em cada exemplo.",
    )
    diagnose_parser = subcommands.add_parser(
        "diagnose",
        help="Inspeciona GPU, CUDA e dependências opcionais de treinamento.",
    )
    diagnose_parser.add_argument(
        "--require-training",
        action="store_true",
        help="Retorna erro quando o ambiente ainda não pode treinar.",
    )
    train_parser = subcommands.add_parser(
        "train",
        help="Executa ou valida uma run de fine-tuning reproduzível.",
    )
    train_parser.add_argument("--config", type=Path, default=settings.training_config_path)
    train_parser.add_argument("--source", type=Path, default=settings.sft_output_path)
    train_parser.add_argument("--output", type=Path, default=settings.training_output_path)
    train_parser.add_argument("--resume-from-checkpoint", type=Path)
    train_parser.add_argument("--producer-commit")
    train_parser.add_argument("--allow-dirty", action="store_true")
    train_parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        settings = Settings.from_environment()
        parser = _build_parser(settings)
        arguments = parser.parse_args(argv)
        if arguments.command == "prepare-sft":
            prepared = prepare_sft_dataset(
                arguments.source,
                arguments.output,
                system_prompt=arguments.system_prompt,
            )
            print(
                json.dumps(
                    prepared.manifest["summary"],
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            print(f"Dataset SFT: {arguments.output.resolve()}")
            return 0
        if arguments.command == "diagnose":
            diagnostic = collect_environment_diagnostic()
            print(json.dumps(diagnostic.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
            if arguments.require_training and not diagnostic.ready_for_training:
                return 2
            return 0
        if arguments.command == "train":
            project_root = Path(__file__).resolve().parents[2]
            outcome = execute_training(
                config_path=arguments.config,
                source_path=arguments.source,
                output_path=arguments.output,
                project_root=project_root,
                resume_from_checkpoint=arguments.resume_from_checkpoint,
                producer_commit=arguments.producer_commit,
                allow_dirty=arguments.allow_dirty,
                dry_run=arguments.dry_run,
            )
            print(json.dumps(outcome.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
            return 0
    except (
        ConfigurationError,
        DatasetReadError,
        SftPreparationError,
        TrainingConfigurationError,
        TrainingDependencyError,
        TrainingExecutionError,
        TrainingPreflightError,
    ) as error:
        print(f"Erro: {error}", file=sys.stderr)
        return 2

    parser.error(f"Comando desconhecido: {arguments.command}")
    return 2
