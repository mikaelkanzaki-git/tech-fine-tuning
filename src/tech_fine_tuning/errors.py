"""Erros estáveis apresentados pela CLI do tech-fine-tuning."""


class ConfigurationError(ValueError):
    """Indica uma configuração ausente ou inválida."""


class DatasetReadError(ValueError):
    """Indica que um artefato de dataset não pôde ser lido."""


class SftPreparationError(ValueError):
    """Indica que o dataset não atende ao contrato de preparação para SFT."""


class TrainingConfigurationError(ValueError):
    """Indica uma configuração de treinamento inválida ou incompatível."""


class TrainingPreflightError(ValueError):
    """Indica que dataset, Git ou ambiente não atendem aos pré-requisitos."""


class TrainingDependencyError(RuntimeError):
    """Indica que o runtime opcional de treinamento não está instalado."""


class TrainingExecutionError(RuntimeError):
    """Indica falha do backend durante uma execução de treinamento."""


class EvaluationPreflightError(ValueError):
    """Indica que modelo, dataset ou destino não permitem uma avaliação válida."""


class EvaluationExecutionError(RuntimeError):
    """Indica uma falha ao gerar ou persistir resultados de avaliação."""


class ReviewValidationError(ValueError):
    """Indica que uma revisão humana está incompleta ou inconsistente."""


class ReviewExecutionError(RuntimeError):
    """Indica uma falha ao persistir o resultado consolidado da revisão."""


class SftAuditValidationError(ValueError):
    """Indica que um dataset SFT não pode ser auditado com segurança."""


class SftAuditExecutionError(RuntimeError):
    """Indica uma falha ao persistir o relatório da auditoria SFT."""


class SftCurationValidationError(ValueError):
    """Indica que o dataset SFT não pode ser curado com segurança."""


class SftCurationExecutionError(RuntimeError):
    """Indica uma falha ao persistir o dataset SFT curado."""
