"""Erros estáveis apresentados pela CLI do tech-fine-tuning."""


class ConfigurationError(ValueError):
    """Indica uma configuração ausente ou inválida."""


class DatasetReadError(ValueError):
    """Indica que um artefato de dataset não pôde ser lido."""


class SftPreparationError(ValueError):
    """Indica que o dataset não atende ao contrato de preparação para SFT."""
