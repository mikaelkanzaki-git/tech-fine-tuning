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
