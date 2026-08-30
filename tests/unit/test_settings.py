from __future__ import annotations

from pathlib import Path

import pytest

from tech_fine_tuning.config.settings import DEFAULT_SYSTEM_PROMPT, Settings
from tech_fine_tuning.errors import ConfigurationError


def test_settings_use_documented_defaults() -> None:
    settings = Settings.from_environment({})

    assert settings.canonical_dataset_path == Path("../tech-ingestao/artifacts/dataset")
    assert settings.sft_output_path == Path("artifacts/sft")
    assert settings.system_prompt == DEFAULT_SYSTEM_PROMPT


def test_settings_read_environment_overrides() -> None:
    settings = Settings.from_environment(
        {
            "TECH_FINE_TUNING_CANONICAL_DATASET_PATH": "input",
            "TECH_FINE_TUNING_SFT_OUTPUT_PATH": "output",
            "TECH_FINE_TUNING_SYSTEM_PROMPT": "Custom system prompt",
        }
    )

    assert settings.canonical_dataset_path == Path("input")
    assert settings.sft_output_path == Path("output")
    assert settings.system_prompt == "Custom system prompt"


def test_settings_reject_empty_system_prompt() -> None:
    with pytest.raises(ConfigurationError, match="não pode ser vazio"):
        Settings(system_prompt=" ")
