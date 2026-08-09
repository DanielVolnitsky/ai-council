import pytest
from pydantic import ValidationError

from core.config import CouncilConfig, ModelConfig, load_config

VALID_YAML = """\
default_synthesizer: openai:gpt-4o

models:
  - id: openai:gpt-4o
    api_key_env: OPENAI_API_KEY
    enabled: true

  - id: anthropic:claude-sonnet-5
    api_key_env: ANTHROPIC_API_KEY
    enabled: false
"""


def test_load_config_returns_valid_config(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(VALID_YAML)

    config = load_config(config_file)

    assert config == CouncilConfig(
        default_synthesizer="openai:gpt-4o",
        models=[
            ModelConfig(
                id="openai:gpt-4o",
                api_key_env="OPENAI_API_KEY",
                enabled=True,
            ),
            ModelConfig(
                id="anthropic:claude-sonnet-5",
                api_key_env="ANTHROPIC_API_KEY",
                enabled=False,
            ),
        ],
    )


def test_enabled_models_excludes_disabled(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(VALID_YAML)

    config = load_config(config_file)

    assert [m.id for m in config.enabled_models] == ["openai:gpt-4o"]


def test_default_synthesizer_not_in_enabled_raises(tmp_path):
    """Without this check the backend starts up fine and fails on the first
    synthesis call instead."""
    yaml_content = """\
default_synthesizer: openai:gpt-4o
models:
  - id: openai:gpt-4o
    enabled: false
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content)

    with pytest.raises(ValidationError, match="default_synthesizer"):
        load_config(config_file)


def test_missing_file_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path/config.yaml")


def test_ollama_model_without_api_key_env_is_valid(tmp_path):
    """Local providers have no API key, so api_key_env=None must validate."""
    yaml_content = """\
default_synthesizer: ollama:llama3
models:
  - id: ollama:llama3
    base_url: http://localhost:11434
    enabled: true
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content)

    config = load_config(config_file)

    assert config == CouncilConfig(
        default_synthesizer="ollama:llama3",
        models=[
            ModelConfig(
                id="ollama:llama3",
                base_url="http://localhost:11434",
                enabled=True,
            ),
        ],
    )
