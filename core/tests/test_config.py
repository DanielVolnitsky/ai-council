# core/tests/test_config.py
#
# Unit tests for the config loader.
#
# Each test writes a YAML file to pytest's `tmp_path` fixture (a per-test
# temporary directory managed by pytest — no manual cleanup needed) and calls
# load_config() on it.
#
# pytest's `tmp_path` fixture is analogous to JUnit 5's @TempDir.

import pytest
from pydantic import ValidationError

from core.config import CouncilConfig, ModelConfig, load_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_YAML = """\
default_synthesizer: openai/gpt-4o

models:
  - id: openai/gpt-4o
    provider: openai
    api_key_env: OPENAI_API_KEY
    enabled: true

  - id: anthropic/claude-3-5-sonnet-20241022
    provider: anthropic
    api_key_env: ANTHROPIC_API_KEY
    enabled: false
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_load_config_returns_valid_config(tmp_path):
    """
    Golden-path: a well-formed config.yaml produces the expected CouncilConfig.

    Asserting against the full expected object catches all fields at once,
    including any newly added ones the test might otherwise silently ignore.
    """
    config_file = tmp_path / "config.yaml"
    config_file.write_text(VALID_YAML)

    config = load_config(config_file)

    assert config == CouncilConfig(
        default_synthesizer="openai/gpt-4o",
        models=[
            ModelConfig(
                id="openai/gpt-4o",
                provider="openai",
                api_key_env="OPENAI_API_KEY",
                enabled=True,
            ),
            ModelConfig(
                id="anthropic/claude-3-5-sonnet-20241022",
                provider="anthropic",
                api_key_env="ANTHROPIC_API_KEY",
                enabled=False,
            ),
        ],
    )


def test_enabled_models_excludes_disabled(tmp_path):
    """enabled_models property only includes models where enabled=true."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(VALID_YAML)

    config = load_config(config_file)

    assert [m.id for m in config.enabled_models] == ["openai/gpt-4o"]


def test_default_synthesizer_not_in_enabled_raises(tmp_path):
    """
    Validation must reject a config where default_synthesizer points to a
    disabled model — otherwise the backend would start up and immediately
    fail on the first synthesis call.
    """
    yaml_content = """\
default_synthesizer: openai/gpt-4o
models:
  - id: openai/gpt-4o
    provider: openai
    enabled: false
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content)

    with pytest.raises(ValidationError, match="default_synthesizer"):
        load_config(config_file)


def test_missing_file_raises_file_not_found():
    """load_config propagates FileNotFoundError for a non-existent path."""
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path/config.yaml")


def test_ollama_model_without_api_key_env_is_valid(tmp_path):
    """
    Local providers (Ollama) have no API key.
    api_key_env=None must not fail validation.
    """
    yaml_content = """\
default_synthesizer: ollama/llama3
models:
  - id: ollama/llama3
    provider: ollama
    base_url: http://localhost:11434
    enabled: true
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content)

    config = load_config(config_file)

    assert config == CouncilConfig(
        default_synthesizer="ollama/llama3",
        models=[
            ModelConfig(
                id="ollama/llama3",
                provider="ollama",
                base_url="http://localhost:11434",
                enabled=True,
            ),
        ],
    )
