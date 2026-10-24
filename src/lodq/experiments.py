"""Command-line runner for the LOD QAOA experiment grid."""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lodq.classical import dp_solve, greedy_solve
from lodq.config import ExperimentConfig, dump_config, load_config
from lodq.evaluate import evaluate_counts, evaluate_probabilities, p_opt_random, random_baseline
from lodq.fastsim import probabilities, qaoa_state
from lodq.ising import qubo_to_ising
from lodq.metrics import problem_from_scene
from lodq.qaoa import (
    QAOAResult,
    build_qaoa_circuit,
    expectation_fn,
    interp_init,
    linear_ramp_init,
    make_estimator,
    normalize_operator,
    run_qaoa,
    run_qaoa_sweep,
    transpile_for_aer,
)
from lodq.qubo import (
    Qubo,
    all_energies,
    build_qubo,
    cache_energies,
    calibrate_penalties,
    energies_cache_path,
    load_energies,
)
from lodq.scene import load_scene
from lodq.storage import load_runs, run_exists, runs_to_dataframe, save_run


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


def _qubo_for_mixer(prep: dict[str, Any], mixer: str) -> Qubo:
    if mixer == "x":
        return prep["qubo"]
    problem = prep["problem"]
    return build_qubo(problem.v, problem.dw, problem.dW, 0.0, prep["penalties"]["B"])


def _solution_dict(solution: Any) -> dict[str, Any]:
    return {
        "choice": list(solution.choice),
        "value": solution.value,
        "used": solution.used,
        "solver": solution.solver,
    }


def _save_result(
    prep: dict[str, Any],
    cfg: ExperimentConfig,
    runs_root: str,
    mixer: str,
    qubo: Qubo,
    result: QAOAResult,
    started: float,
) -> None:
    problem = prep["problem"]
    baseline = p_opt_random(qubo)
    metrics = evaluate_counts(result.counts, qubo, problem.v, problem.dw, prep["dp"], baseline)
    metrics_exact = None
    if cfg.backend == "fastsim":
        energies = prep["energies"] if qubo is prep["qubo"] else all_energies(qubo)
        state = qaoa_state(energies / result.scale, result.gammas, result.betas, qubo.n)
        distribution = probabilities(state)
        metrics_exact = evaluate_probabilities(
            distribution, qubo, problem.v, problem.dw, prep["dp"], baseline
        ).to_dict()
    payload = result.to_dict()
    payload["mixer"] = mixer
    save_run(
        {
            "config": dump_config(cfg),
            "scene": prep["name"],
            "problem": {"v": problem.v, "dw": problem.dw, "dW": problem.dW},
            "penalties": {"A": qubo.A, "B": qubo.B},
            "dp": _solution_dict(prep["dp"]),
            "greedy": _solution_dict(prep["greedy"]),
            "random": prep["random"].to_dict(),
            "qaoa": payload,
            "metrics": metrics.to_dict(),
            "metrics_exact": metrics_exact,
            "timings": {
                "wall_time_s": result.wall_time_s,
                "n_evals": result.n_evals,
                "sweep_s": time.perf_counter() - started,
            },
        },
        tag=f"{prep['name']}__{mixer}__s{result.seed}__p{result.p}",
        root=runs_root,
    )


def _tag(scene: str, mixer: str, seed: int, depth: int) -> str:
    return f"{scene}__{mixer}__s{seed}__p{depth}"


def _saved_angles(runs_root: str, tag: str) -> tuple[np.ndarray, np.ndarray] | None:
    for run in load_runs(runs_root):
        if run.get("tag") != tag:
            continue
        qaoa = run["qaoa"]
        return (
            np.asarray(qaoa["gammas"], dtype=np.float64),
            np.asarray(qaoa["betas"], dtype=np.float64),
        )
    return None


def _grow_angles(
    gammas: np.ndarray, betas: np.ndarray, depth: int
) -> tuple[np.ndarray, np.ndarray]:
    gammas = np.asarray(gammas, dtype=np.float64)
    betas = np.asarray(betas, dtype=np.float64)
    while gammas.size < depth:
        gammas, betas = interp_init(gammas, betas)
    return gammas, betas


def _sweep_kwargs(cfg: ExperimentConfig) -> dict[str, Any]:
    return {
        "n_starts": cfg.n_starts,
        "maxiter": cfg.maxiter,
        "shots": cfg.shots,
        "objective": cfg.objective,
        "cvar_alpha": cfg.cvar_alpha,
        "backend": cfg.backend,
    }


def run_scene(prep: dict[str, Any], cfg: ExperimentConfig, runs_root: str) -> None:
    """Sweep mixers, seeds, and p, skipping combinations that are already stored."""
    for mixer in cfg.mixers:
        qubo = _qubo_for_mixer(prep, mixer)
        for seed in cfg.seeds:
            tags = [_tag(prep["name"], mixer, seed, depth) for depth in cfg.p_list]
            if all(run_exists(runs_root, tag) for tag in tags):
                print(f"skip {prep['name']} {mixer} s{seed}")
                continue
            started = time.perf_counter()
            missing = [
                depth
                for depth, tag in zip(cfg.p_list, tags, strict=True)
                if not run_exists(runs_root, tag)
            ]
            if len(missing) == len(cfg.p_list):
                results = run_qaoa_sweep(qubo, list(cfg.p_list), seed, **_sweep_kwargs(cfg))
                for result in results:
                    _save_result(prep, cfg, runs_root, mixer, qubo, result, started)
                continue
            warm: tuple[np.ndarray, np.ndarray] | None = None
            for depth, tag in zip(cfg.p_list, tags, strict=True):
                if run_exists(runs_root, tag):
                    loaded = _saved_angles(runs_root, tag)
                    if loaded is not None:
                        warm = loaded
                    continue
                warm_start = None if warm is None else _grow_angles(*warm, depth)
                result = run_qaoa(qubo, depth, seed, warm_start=warm_start, **_sweep_kwargs(cfg))
                _save_result(prep, cfg, runs_root, mixer, qubo, result, started)
                warm = (result.gammas, result.betas)


def estimate_time(prep: dict[str, Any], cfg: ExperimentConfig) -> float:
    """Seconds for this scene: three <H> evaluations at max p, scaled by the grid."""
    qubo = prep["qubo"]
    depth = max(cfg.p_list)
    operator, _const = qubo_to_ising(qubo)
    cost, _scale = normalize_operator(operator)
    circuit = transpile_for_aer(build_qaoa_circuit(cost, depth), 0)
    objective = expectation_fn(circuit, cost, make_estimator(cfg.backend, seed=0), slow_eval_s=1e9)
    gammas, betas = linear_ramp_init(depth)
    point = np.concatenate((gammas, betas))
    samples = []
    for _ in range(3):
        started = time.perf_counter()
        objective(point)
        samples.append(time.perf_counter() - started)
    mean = sum(samples) / len(samples)
    grid = cfg.maxiter * cfg.n_starts * len(cfg.seeds) * len(cfg.mixers) * len(cfg.p_list)
    return mean * grid


def write_summary(runs_root: str) -> Path:
    """Average and standard deviation of metric columns, grouped by scene, mixer, and p."""
    frame = runs_to_dataframe(load_runs(runs_root))
    path = Path(runs_root) / "summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        path.write_text("", encoding="utf-8")
        return path
    keys = ["scene", "qaoa.mixer", "qaoa.p"]
    value_cols = [
        column
        for column in frame.columns
        if column.startswith(("metrics.", "metrics_exact."))
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    summary = frame.groupby(keys, dropna=False)[value_cols].agg(["mean", "std"])
    summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
    summary.reset_index().to_csv(path, index=False)
    return path


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
        total = 0.0
        for name in scenes:
            seconds = estimate_time(prepare_scene(name, cfg), cfg)
            total += seconds
            print(f"{name} estimated {seconds:.1f} s")
        print(f"estimated total {total:.1f} s")
        return
    for name in scenes:
        run_scene(prepare_scene(name, cfg), cfg, args.runs_root)
    write_summary(args.runs_root)


if __name__ == "__main__":
    main()
