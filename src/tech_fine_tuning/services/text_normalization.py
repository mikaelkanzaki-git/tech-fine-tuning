"""Normalização textual compartilhada pelas etapas de qualidade do SFT."""

from __future__ import annotations

import re
import unicodedata


def normalize_text(value: str) -> str:
    """Produz uma chave estável sem reescrever o texto persistido."""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = " ".join(normalized.split())
    return re.sub(r"\s+([?!.,;:])", r"\1", normalized)
