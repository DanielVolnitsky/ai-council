# core/core/config.py
#
# Loads config.yaml into a validated, typed Python object.
#
# Why Pydantic for config?
#   config.yaml is an external input (like an API request body) — it can be
#   malformed, missing required fields, or structurally wrong.  Pydantic's
#   model_validate() gives us free field presence checks, type coercion, and
#   clear error messages, replacing what would be a hand-rolled validation loop.
#
#   Java analogy: parsing config.yaml into a @ConfigurationProperties class
#   annotated with @Validated, where Spring Boot raises a startup error if any
#   required field is absent or has the wrong type.

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, model_validator


class ModelConfig(BaseModel):
    """
    Configuration for one LLM model as declared in config.yaml.

    `id` is the unique key used throughout the system (e.g. "openai/gpt-4o").
    LiteLLM uses this exact "<provider>/<model>" format for routing, so the
    config id doubles as the LiteLLM model name — no translation needed.

    `api_key_env` is the *name* of the environment variable that holds the
    API key (e.g. "OPENAI_API_KEY"), not the key itself.  The backend reads
    os.environ[model.api_key_env] at call time.  This indirection keeps
    secrets out of config files.

    `base_url` is only needed for self-hosted providers like Ollama.
    """
    id: str
    provider: str
    api_key_env: str | None = None   # None for local providers (Ollama)
    base_url: str | None = None      # Override the provider's default endpoint
    enabled: bool = True


class CouncilConfig(BaseModel):
    """
    The full configuration loaded from config.yaml.

    `default_synthesizer` must be the id of an enabled model.  The
    `_synthesizer_must_be_enabled` validator enforces this at load time so
    the backends can assume it is always valid.

    `enabled_models` is a @property (not a stored field) so it is always
    computed from the current `models` list rather than being a stale snapshot.
    Pydantic does not serialise properties — only `BaseModel` fields are
    included in model_dump() output.
    """
    default_synthesizer: str
    models: list[ModelConfig]

    @property
    def enabled_models(self) -> list[ModelConfig]:
        """Convenience filter: only models where enabled=true."""
        return [m for m in self.models if m.enabled]

    @model_validator(mode="after")
    def _synthesizer_must_be_enabled(self) -> "CouncilConfig":
        """
        Fail fast at startup if default_synthesizer is not in the enabled list.

        `mode="after"` means this validator runs after all fields have been
        populated and type-checked.  `self` is the fully constructed model
        instance, so we can call self.enabled_models here.

        Raising ValueError inside a Pydantic validator causes model_validate()
        to raise a ValidationError, which is what callers expect.
        """
        enabled_ids = {m.id for m in self.enabled_models}
        if self.default_synthesizer not in enabled_ids:
            raise ValueError(
                f"default_synthesizer '{self.default_synthesizer}' is not in the "
                f"enabled models list.  Enabled model IDs: {sorted(enabled_ids)}"
            )
        return self


def load_config(path: str | Path = "config.yaml") -> CouncilConfig:
    """
    Read config.yaml from `path` and return a validated CouncilConfig.

    Raises:
        FileNotFoundError   — if config.yaml does not exist.
        yaml.YAMLError      — if the file is not valid YAML.
        pydantic.ValidationError — if required fields are missing or the
                              default_synthesizer is not in the enabled list.

    All three are programming / deployment errors, not runtime-recoverable
    states, so they are allowed to propagate to the caller (the backend
    startup sequence) which will crash-and-log them.

    `yaml.safe_load` is used instead of `yaml.load` to prevent arbitrary
    Python object deserialisation from untrusted YAML.
    """
    raw: dict = yaml.safe_load(Path(path).read_text())
    # model_validate() accepts a plain dict and returns a fully validated
    # CouncilConfig instance.  Equivalent to Jackson's objectMapper.readValue().
    return CouncilConfig.model_validate(raw)
