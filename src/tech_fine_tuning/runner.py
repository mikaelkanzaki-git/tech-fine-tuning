"""Interface de linha de comando do serviço de fine-tuning."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from tech_fine_tuning.config.settings import Settings
from tech_fine_tuning.errors import ConfigurationError, DatasetReadError, SftPreparationError
from tech_fine_tuning.services.sft_dataset_service import prepare_sft_dataset


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
    except (ConfigurationError, DatasetReadError, SftPreparationError) as error:
        print(f"Erro: {error}", file=sys.stderr)
        return 2

    parser.error(f"Comando desconhecido: {arguments.command}")
    return 2
