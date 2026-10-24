"""Command-line runner for the LOD QAOA experiment grid."""

import argparse
import json
from pathlib import Path
from typing import Any

from lodq.classical import dp_solve, greedy_solve
from lodq.config import ExperimentConfig, load_config
from lodq.evaluate import random_baseline
from lodq.metrics import problem_from_scene
from lodq.qubo import (
    build_qubo,
    cache_energies,
    calibrate_penalties,
    energies_cache_path,
    load_energies,
)
from lodq.scene import load_scene


def _penalties(name: str, problem: Any, cfg: ExperimentConfig, dp: Any) -> tuple[float, float]:
    auto_a = cfg.penalty_A == "auto"
    auto_b = cfg.penalty_B == "auto"
    path = Path("cache") / f"{name}_penalties.json"
    if auto_a and auto_b and path.is_file():
        saved = json.loads(path.read_text(encoding="utf-8"))
        return float(saved["A"]), float(saved["B"])
    if auto_a or auto_b:
        penalty_a, penalty_b, _report = calibrate_penalties(problem.v, problem.dw, problem.dW, dp)
    else:
        penalty_a, penalty_b = float(cfg.penalty_A), float(cfg.penalty_B)
    if not auto_a:
        penalty_a = float(cfg.penalty_A)
    if not auto_b:
        penalty_b = float(cfg.penalty_B)
    if auto_a and auto_b:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"A": penalty_a, "B": penalty_b}), encoding="utf-8")
    return penalty_a, penalty_b


def _energies(name: str, qubo: Any) -> Any:
    path = energies_cache_path(name, qubo)
    if path.is_file():
        return load_energies(path)
    cache_energies(qubo, path)
    return load_energies(path)


def prepare_scene(name: str, cfg: ExperimentConfig) -> dict[str, Any]:
    """Load one scene, solve the classical baselines, and build its QUBO."""
    scene = load_scene(f"scenes/{name}.yaml")
    problem = problem_from_scene(scene)
    dp = dp_solve(problem.v, problem.dw, problem.dW)
    greedy = greedy_solve(problem.v, problem.dw, problem.dW)
    penalty_a, penalty_b = _penalties(name, problem, cfg, dp)
    qubo = build_qubo(problem.v, problem.dw, problem.dW, penalty_a, penalty_b)
    return {
        "name": name,
        "problem": problem,
        "dp": dp,
        "greedy": greedy,
        "random": random_baseline(qubo, problem.v, problem.dw, dp),
        "penalties": {"A": penalty_a, "B": penalty_b},
        "qubo": qubo,
        "energies": _energies(name, qubo),
    }


def shard_scenes(scenes: list[str], shard: str | None) -> list[str]:
    """Keep every n-th scene, starting at offset i, for a shard written as i/n."""
    names = list(scenes)
    if shard is None:
        return names
    try:
        index_text, count_text = shard.split("/")
        index, count = int(index_text), int(count_text)
    except ValueError as exc:
        raise ValueError(f"shard must look like i/n, got {shard!r}") from exc
    if count < 1 or not 0 <= index < count:
        raise ValueError(f"shard i/n must satisfy 0 <= i < n, got {shard}")
    return names[index::count]


def _jobs(scenes: list[str], cfg: ExperimentConfig) -> list[tuple[str, str, int, int]]:
    return [
        (scene, mixer, seed, p)
        for scene in scenes
        for mixer in cfg.mixers
        for seed in cfg.seeds
        for p in cfg.p_list
    ]


def run_scene(prep: dict[str, Any], cfg: ExperimentConfig) -> None:
    """Run QAOA for one prepared scene. Filled in by the next task."""
    del prep, cfg


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the LOD QAOA experiment grid")
    parser.add_argument("--config", required=True)
    parser.add_argument("--shard", default=None, help="shard i/n, for example 0/5")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--runs-root", default="runs")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.quick:
        cfg = cfg.model_copy(update={"p_list": [1], "seeds": [0], "maxiter": 50, "n_starts": 1})
    scenes = shard_scenes(list(cfg.scene_presets), args.shard)
    if args.dry_run:
        jobs = _jobs(scenes, cfg)
        for scene, mixer, seed, depth in jobs:
            print(f"{scene} {mixer} {seed} {depth}")
        print(f"runs {len(jobs)}")
        return
    for name in scenes:
        run_scene(prepare_scene(name, cfg), cfg)


if __name__ == "__main__":
    main()
