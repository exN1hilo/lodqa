import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from lodq.config import ExperimentConfig, dump_config, load_config

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"
CONFIG_PATHS = sorted([*CONFIGS_DIR.glob("*.yaml"), *CONFIGS_DIR.glob("*.yml")])


def make(**kw) -> ExperimentConfig:
    return ExperimentConfig(name="test", scene_presets=["default"], **kw)


def test_defaults() -> None:
    cfg = make()
    assert cfg.p_list == [1, 2, 3, 4]
    assert cfg.mixers == ["x"]
    assert cfg.seeds == [0, 1, 2, 3, 4]
    assert cfg.optimizer == "COBYLA"
    assert cfg.maxiter == 300
    assert cfg.n_starts == 5
    assert cfg.shots == 4096
    assert cfg.objective == "energy"
    assert cfg.cvar_alpha == 0.1
    assert cfg.penalty_A == "auto"
    assert cfg.penalty_B == "auto"
    assert cfg.backend == "aer"


def test_list_defaults_not_shared() -> None:
    a = make()
    a.p_list.append(5)
    assert make().p_list == [1, 2, 3, 4]


def test_extra_field_forbidden() -> None:
    with pytest.raises(ValidationError):
        make(unknown=1)


def test_required_fields() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig(scene_presets=["default"])
    with pytest.raises(ValidationError):
        ExperimentConfig(name="test")


@pytest.mark.parametrize("alpha", [1e-6, 0.5, 1.0])
def test_cvar_alpha_valid(alpha: float) -> None:
    assert make(cvar_alpha=alpha).cvar_alpha == alpha


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.01])
def test_cvar_alpha_invalid(alpha: float) -> None:
    with pytest.raises(ValidationError):
        make(cvar_alpha=alpha)


@pytest.mark.parametrize(
    "kw",
    [
        {"p_list": []},
        {"p_list": [0]},
        {"p_list": [1, -2]},
        {"seeds": []},
        {"seeds": [-1]},
        {"shots": 0},
        {"maxiter": 0},
        {"n_starts": 0},
        {"scene_presets": []},
        {"mixers": []},
        {"mixers": ["z"]},
        {"optimizer": "SPSA"},
        {"objective": "max"},
        {"backend": "gpu"},
    ],
)
def test_invalid_values(kw: dict) -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig(**{"name": "test", "scene_presets": ["default"], **kw})


@pytest.mark.parametrize("value", ["auto", 2.5, 3])
def test_penalty_valid(value: float | str) -> None:
    cfg = make(penalty_A=value, penalty_B=value)
    assert cfg.penalty_A == value
    assert cfg.penalty_B == value


@pytest.mark.parametrize("value", ["manual", 0, -1.0])
def test_penalty_invalid(value: float | str) -> None:
    with pytest.raises(ValidationError):
        make(penalty_A=value)
    with pytest.raises(ValidationError):
        make(penalty_B=value)


def test_load_config(tmp_path: Path) -> None:
    path = tmp_path / "exp.yaml"
    path.write_text(
        "name: smoke\n"
        "scene_presets: [default]\n"
        "p_list: [1]\n"
        "seeds: [0]\n"
        "mixers: [x, xy]\n"
        "maxiter: 50\n"
        "penalty_B: 0.75\n"
        "backend: fastsim\n",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.name == "smoke"
    assert cfg.p_list == [1]
    assert cfg.mixers == ["x", "xy"]
    assert cfg.maxiter == 50
    assert cfg.penalty_A == "auto"
    assert cfg.penalty_B == 0.75
    assert cfg.backend == "fastsim"
    assert load_config(str(path)) == cfg


def test_load_config_rejects_unknown_key(tmp_path: Path) -> None:
    path = tmp_path / "exp.yaml"
    path.write_text("name: smoke\nscene_presets: [default]\nshot: 10\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(path)


@pytest.mark.parametrize("text", ["", "- a\n- b\n"])
def test_load_config_not_a_mapping(tmp_path: Path, text: str) -> None:
    path = tmp_path / "exp.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(TypeError, match="YAML-словарь"):
        load_config(path)


def test_dump_config_roundtrip() -> None:
    cfg = make(p_list=[1, 2], mixers=["x", "xy"], penalty_A=1.5, objective="cvar")
    data = dump_config(cfg)
    assert isinstance(data, dict)
    assert data["name"] == "test"
    assert data["penalty_A"] == 1.5
    assert data["penalty_B"] == "auto"
    assert json.loads(json.dumps(data)) == data
    assert ExperimentConfig.model_validate(data) == cfg


def test_configs_dir_not_empty() -> None:
    assert CONFIG_PATHS, f"в {CONFIGS_DIR} нет ни одного YAML-конфига"


@pytest.mark.parametrize("path", CONFIG_PATHS, ids=lambda p: p.name)
def test_all_configs_load(path: Path) -> None:
    cfg = load_config(path)
    assert ExperimentConfig.model_validate(dump_config(cfg)) == cfg


def test_smoke_config() -> None:
    cfg = load_config(CONFIGS_DIR / "smoke.yaml")
    assert cfg.name == "smoke"
    assert cfg.scene_presets == ["default"]
    assert cfg.p_list == [1]
    assert cfg.seeds == [0]
    assert cfg.maxiter == 50
