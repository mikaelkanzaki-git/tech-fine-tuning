from __future__ import annotations

from pathlib import Path

import pytest

from tech_fine_tuning.errors import TrainingConfigurationError
from tech_fine_tuning.integrations.training.config_reader import read_training_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SMOKE_CONFIG = PROJECT_ROOT / "configs" / "qwen3-4b" / "smoke.toml"
FULL_CONFIG = PROJECT_ROOT / "configs" / "qwen3-4b" / "full.toml"


def _changed_config(tmp_path: Path, old: str, new: str) -> Path:
    text = SMOKE_CONFIG.read_text(encoding="utf-8")
    assert old in text
    path = tmp_path / "training.toml"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return path


def test_reads_smoke_and_full_profiles() -> None:
    smoke = read_training_config(SMOKE_CONFIG)
    full = read_training_config(FULL_CONFIG)

    assert smoke.run_name == "qwen3-4b-medquad-smoke"
    assert smoke.model.load_in_4bit is True
    assert smoke.lora.target_modules[0] == "q_proj"
    assert smoke.dataset.train_limit == 1000
    assert smoke.trainer.max_steps == 50
    assert full.dataset.train_limit is None
    assert full.trainer.num_train_epochs == 1.0
    assert smoke.as_dict()["model"]["base_id"] == "Qwen/Qwen3-4B-Instruct-2507"


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('schema_version = "1.0"', 'schema_version = "2.0"', "não suportado"),
        ('run_name = "qwen3-4b-medquad-smoke"', 'run_name = ""', "não vazia"),
        (
            'revision = "7744afa8566e264af1a92a806d8d9aae00cc7c78"',
            'revision = "main"',
            "revisão Git imutável",
        ),
        ("max_sequence_length = 2048", "max_sequence_length = 0", "deve ser positivo"),
        ("load_in_4bit = true", 'load_in_4bit = "yes"', "deve ser booleano"),
        ("rank = 16", "rank = 0", "rank/alpha"),
        ("dropout = 0.0", 'dropout = "zero"', "deve ser numérico"),
        (
            "target_modules = ["
            '"q_proj", "k_proj", "v_proj", "o_proj", '
            '"gate_proj", "up_proj", "down_proj"]',
            "target_modules = []",
            "deve ser uma lista",
        ),
        ("train_limit = 1000", "train_limit = 0", "deve ser positivo"),
        ("num_proc = 2", "num_proc = 0", "num_proc deve ser positivo"),
        (
            "max_steps = 50",
            "max_steps = 50\nnum_train_epochs = 1.0",
            "exatamente um",
        ),
        ("max_steps = 50", "num_train_epochs = 0.0", "deve ser positivo"),
        ("learning_rate = 0.0002", "learning_rate = 0.0", "fora dos limites"),
        ("warmup_steps = 5", "warmup_steps = -1", "fora dos limites"),
    ],
)
def test_rejects_invalid_profile_values(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    path = _changed_config(tmp_path, old, new)

    with pytest.raises(TrainingConfigurationError, match=message):
        read_training_config(path)


def test_rejects_unknown_and_missing_sections(tmp_path: Path) -> None:
    unknown = _changed_config(
        tmp_path,
        'schema_version = "1.0"',
        'schema_version = "1.0"\nextra = 1',
    )
    with pytest.raises(TrainingConfigurationError, match="campos desconhecidos"):
        read_training_config(unknown)

    missing = tmp_path / "missing-section.toml"
    missing.write_text('schema_version = "1.0"\nrun_name = "run"\n', encoding="utf-8")
    with pytest.raises(TrainingConfigurationError, match="seção 'model' é obrigatória"):
        read_training_config(missing)


def test_reports_missing_and_invalid_toml(tmp_path: Path) -> None:
    with pytest.raises(TrainingConfigurationError, match="Não foi possível ler"):
        read_training_config(tmp_path / "missing.toml")

    invalid = tmp_path / "invalid.toml"
    invalid.write_text("not = [valid", encoding="utf-8")
    with pytest.raises(TrainingConfigurationError, match="Não foi possível ler"):
        read_training_config(invalid)
