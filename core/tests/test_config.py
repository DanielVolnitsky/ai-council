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


def test_api_key_is_read_from_the_named_env_var(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")

    assert ModelConfig(id="openai:gpt-4o", api_key_env="OPENAI_API_KEY").api_key == "sk-test-123"


def test_api_key_is_none_without_an_env_var_name():
    assert ModelConfig(id="ollama:llama3").api_key is None


def test_missing_api_key_env_var_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(KeyError, match="OPENAI_API_KEY"):
        ModelConfig(id="openai:gpt-4o", api_key_env="OPENAI_API_KEY").api_key


def test_ollama_model_without_api_key_env_is_valid(tmp_path):
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
