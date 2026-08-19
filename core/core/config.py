from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, model_validator


class ModelConfig(BaseModel):
    id: str
    api_key_env: str | None = None
    base_url: str | None = None
    enabled: bool = True


class CouncilConfig(BaseModel):
    default_synthesizer: str
    models: list[ModelConfig]

    @property
    def enabled_models(self) -> list[ModelConfig]:
        return [m for m in self.models if m.enabled]

    @model_validator(mode="after")
    def _synthesizer_must_be_enabled(self) -> "CouncilConfig":
        enabled_ids = {m.id for m in self.enabled_models}
        if self.default_synthesizer not in enabled_ids:
            raise ValueError(
                f"default_synthesizer '{self.default_synthesizer}' is not in the "
                f"enabled models list.  Enabled model IDs: {sorted(enabled_ids)}"
            )
        return self


def load_config(path: str | Path = "config.yaml") -> CouncilConfig:
    raw: dict = yaml.safe_load(Path(path).read_text())
    return CouncilConfig.model_validate(raw)
