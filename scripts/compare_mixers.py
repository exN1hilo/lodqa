import argparse
from dataclasses import asdict
from pathlib import Path
from statistics import fmean

from lodq.classical import dp_solve
from lodq.evaluate import Metrics, evaluate_counts, p_opt_random
from lodq.metrics import problem_from_scene
from lodq.qaoa import run_qaoa_sweep, save_qaoa_run
from lodq.qubo import build_qubo, calibrate_penalty_B_onehot, default_penalties
from lodq.scene import load_scene


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare X and XY QAOA mixers on one scene")
    parser.add_argument("--scene", type=Path, default=Path("greedy_trap_1"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--p", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--n-starts", type=int, default=5)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--backend", choices=["aer", "statevector", "fastsim"], default="aer")
    parser.add_argument("--root", type=Path, default=Path("runs"))
    args = parser.parse_args()

    scene_path = args.scene
    if not scene_path.suffix:
        scene_path = Path("scenes") / scene_path.with_suffix(".yaml")
    name = scene_path.stem
    problem = problem_from_scene(load_scene(scene_path))
    dp = dp_solve(problem.v, problem.dw, problem.dW)
    A, B = default_penalties(problem.v)
    B_xy, _ = calibrate_penalty_B_onehot(problem.v, problem.dw, problem.dW, dp)
    qubos = {
        "x": build_qubo(problem.v, problem.dw, problem.dW, A, B),
        "xy": build_qubo(problem.v, problem.dw, problem.dW, 0.0, B_xy),
    }
    summary: dict[tuple[int, str], list[Metrics]] = {}
    print(f"scene {name}: DP {dp.choice}, V_dp = {dp.value:.6f}", flush=True)
    print(f"X: A = {A:.6f}, B = {B:.6f}; XY: A = 0, B = {B_xy:.6f}", flush=True)
    print(f"{'p':>2} {'mixer':>5} {'seed':>4} {'p_onehot':>10} {'p_feas':>10} {'ar_best':>10}  run")

    for mixer, qubo in qubos.items():
        for seed in args.seeds:
            print(f"Running {mixer}, seed={seed}, p={args.p}", flush=True)
            results = run_qaoa_sweep(
                qubo,
                args.p,
                seed,
                mixer=mixer,
                n_starts=args.n_starts,
                maxiter=args.maxiter,
                shots=args.shots,
                backend=args.backend,
            )
            for result in results:
                metrics = evaluate_counts(
                    result.counts, qubo, problem.v, problem.dw, dp, p_opt_random(qubo)
                )
                path = save_qaoa_run(
                    result,
                    name,
                    root=args.root,
                    extra={
                        "problem": {"v": problem.v, "dw": problem.dw, "dW": problem.dW},
                        "penalties": {"A": qubo.A, "B": qubo.B},
                        "dp": asdict(dp),
                        "metrics": metrics.to_dict(),
                    },
                )
                summary.setdefault((result.p, mixer), []).append(metrics)
                print(
                    f"{result.p:2d} {mixer:>5} {seed:4d} {metrics.p_onehot:10.4f} "
                    f"{metrics.p_feas:10.4f} {metrics.ar_best:10.4f}  {path}",
                    flush=True,
                )

    print("\nMean over seeds (ar_best = best feasible value / V_dp)")
    print(f"{'p':>2} {'mixer':>5} {'p_onehot':>10} {'p_feas':>10} {'ar_best':>10}")
    for p in args.p:
        for mixer in qubos:
            rows = summary[p, mixer]
            print(
                f"{p:2d} {mixer:>5} {fmean(row.p_onehot for row in rows):10.4f} "
                f"{fmean(row.p_feas for row in rows):10.4f} "
                f"{fmean(row.ar_best for row in rows):10.4f}"
            )


if __name__ == "__main__":
    main()
