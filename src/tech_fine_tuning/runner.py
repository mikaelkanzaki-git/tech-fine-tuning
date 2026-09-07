"""Interface de linha de comando do serviço de fine-tuning."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from tech_fine_tuning.config.settings import Settings
from tech_fine_tuning.errors import (
    ConfigurationError,
    DatasetReadError,
    EvaluationExecutionError,
    EvaluationPreflightError,
    ReviewExecutionError,
    ReviewValidationError,
    SftAuditExecutionError,
    SftAuditValidationError,
    SftCurationExecutionError,
    SftCurationValidationError,
    SftPreparationError,
    TrainingConfigurationError,
    TrainingDependencyError,
    TrainingExecutionError,
    TrainingPreflightError,
)
from tech_fine_tuning.integrations.system.diagnostics import collect_environment_diagnostic
from tech_fine_tuning.models.evaluation import EvaluationSplit
from tech_fine_tuning.services.evaluation_service import execute_evaluation
from tech_fine_tuning.services.review_service import execute_review_summary
from tech_fine_tuning.services.sft_audit_service import audit_sft_dataset
from tech_fine_tuning.services.sft_curation_service import curate_sft_dataset
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
    evaluate_parser = subcommands.add_parser(
        "evaluate",
        help="Compara o adaptador treinado com o modelo base em dados reservados.",
    )
    evaluate_parser.add_argument(
        "--manifest",
        type=Path,
        default=settings.training_output_path / "model-manifest.json",
        help="Contrato model-manifest.json gerado pelo treinamento.",
    )
    evaluate_parser.add_argument(
        "--source",
        type=Path,
        default=settings.sft_output_path,
        help="Diretório do dataset SFT e seu manifesto.",
    )
    evaluate_parser.add_argument(
        "--output",
        type=Path,
        help="Diretório de saída; por padrão, <evaluation-output>/<split>.",
    )
    evaluate_parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
        help="Use validation durante a iteração e reserve test para a decisão final.",
    )
    evaluate_parser.add_argument("--sample-size", type=int, default=20)
    evaluate_parser.add_argument("--seed", type=int, default=3407)
    evaluate_parser.add_argument("--max-new-tokens", type=int, default=256)
    evaluate_parser.add_argument(
        "--compare-base",
        action="store_true",
        help="Também gera respostas do modelo base para comparação lado a lado.",
    )
    review_parser = subcommands.add_parser(
        "summarize-review",
        help="Valida a revisão humana e registra a decisão do quality gate.",
    )
    review_parser.add_argument(
        "--review",
        type=Path,
        required=True,
        help="Arquivo human-review.csv preenchido.",
    )
    review_parser.add_argument(
        "--evaluation-manifest",
        type=Path,
        required=True,
        help="Manifesto da avaliação que gerou a planilha.",
    )
    review_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Diretório novo ou vazio para a revisão consolidada.",
    )
    audit_parser = subcommands.add_parser(
        "audit-sft",
        help="Audita qualidade textual, comprimentos e isolamento dos splits SFT.",
    )
    audit_parser.add_argument(
        "--source",
        type=Path,
        default=settings.sft_output_path,
        help="Diretório do dataset SFT e seu manifesto.",
    )
    audit_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Diretório novo ou vazio para o relatório versionável.",
    )
    audit_parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=2048,
        help="Limite usado para sinalizar provável estouro de contexto.",
    )
    curate_parser = subcommands.add_parser(
        "curate-sft",
        help="Aplica a curadoria conservadora e gera um novo dataset SFT.",
    )
    curate_parser.add_argument(
        "--source",
        type=Path,
        default=settings.sft_output_path,
        help="Diretório do dataset SFT auditado.",
    )
    curate_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Diretório novo ou vazio para o dataset curado.",
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
        if arguments.command == "diagnose":
            diagnostic = collect_environment_diagnostic()
            print(json.dumps(diagnostic.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
            if arguments.require_training and not diagnostic.ready_for_training:
                return 2
            return 0
        if arguments.command == "train":
            project_root = Path(__file__).resolve().parents[2]
            training_outcome = execute_training(
                config_path=arguments.config,
                source_path=arguments.source,
                output_path=arguments.output,
                project_root=project_root,
                resume_from_checkpoint=arguments.resume_from_checkpoint,
                producer_commit=arguments.producer_commit,
                allow_dirty=arguments.allow_dirty,
                dry_run=arguments.dry_run,
            )
            print(
                json.dumps(
                    training_outcome.as_dict(), ensure_ascii=False, indent=2, sort_keys=True
                )
            )
            return 0
        if arguments.command == "evaluate":
            split = cast(EvaluationSplit, arguments.split)
            output_path = arguments.output or settings.evaluation_output_path / split
            evaluation_outcome = execute_evaluation(
                model_manifest_path=arguments.manifest,
                dataset_path=arguments.source,
                output_path=output_path,
                split=split,
                sample_size=arguments.sample_size,
                seed=arguments.seed,
                compare_base=arguments.compare_base,
                max_new_tokens=arguments.max_new_tokens,
            )
            print(
                json.dumps(
                    evaluation_outcome.as_dict(),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if arguments.command == "summarize-review":
            review_outcome = execute_review_summary(
                review_path=arguments.review,
                evaluation_manifest_path=arguments.evaluation_manifest,
                output_path=arguments.output,
            )
            print(
                json.dumps(
                    review_outcome.as_dict(),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if arguments.command == "audit-sft":
            audit_outcome = audit_sft_dataset(
                source_path=arguments.source,
                output_path=arguments.output,
                max_sequence_length=arguments.max_sequence_length,
            )
            print(
                json.dumps(
                    audit_outcome.as_dict(),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if arguments.command == "curate-sft":
            curation_outcome = curate_sft_dataset(
                source_path=arguments.source,
                output_path=arguments.output,
            )
            print(
                json.dumps(
                    curation_outcome.as_dict(),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
    except (
        ConfigurationError,
        DatasetReadError,
        EvaluationExecutionError,
        EvaluationPreflightError,
        ReviewExecutionError,
        ReviewValidationError,
        SftAuditExecutionError,
        SftAuditValidationError,
        SftCurationExecutionError,
        SftCurationValidationError,
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
