from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

PositiveInt = Annotated[int, Field(ge=1)]
Seed = Annotated[int, Field(ge=0)]
Penalty = Annotated[float, Field(gt=0)] | Literal["auto"]


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    scene_presets: list[str] = Field(min_length=1)
    p_list: list[PositiveInt] = Field(default=[1, 2, 3, 4], min_length=1)
    mixers: list[Literal["x", "xy"]] = Field(default=["x"], min_length=1)
    seeds: list[Seed] = Field(default=[0, 1, 2, 3, 4], min_length=1)
    optimizer: Literal["COBYLA"] = "COBYLA"
    maxiter: PositiveInt = 300
    n_starts: PositiveInt = 5
    shots: PositiveInt = 4096
    objective: Literal["energy", "cvar"] = "energy"
    cvar_alpha: float = Field(default=0.1, gt=0, le=1)
    penalty_A: Penalty = "auto"
    penalty_B: Penalty = "auto"
    backend: Literal["aer", "fastsim"] = "aer"


def load_config(path: str | Path) -> ExperimentConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected a YAML mapping with ExperimentConfig fields")
    return ExperimentConfig.model_validate(data)


def dump_config(cfg: ExperimentConfig) -> dict[str, Any]:
    return cfg.model_dump(mode="json")
