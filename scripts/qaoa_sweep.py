"""Run a warm-started QAOA sweep over p on one scene and save every run to runs/."""

import argparse
from pathlib import Path

from lodq.classical import dp_solve
from lodq.metrics import problem_from_scene
from lodq.qaoa import run_qaoa_sweep, save_qaoa_run
from lodq.qubo import bits_from_bitstring, build_qubo, decode, default_penalties, is_feasible
from lodq.scene import load_scene


def main() -> None:
    parser = argparse.ArgumentParser(description="Warm-started QAOA sweep p = 1..4 on one scene")
    parser.add_argument("--scene", type=Path, default=Path("scenes/default.yaml"))
    parser.add_argument("--p", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-starts", type=int, default=5)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--objective", choices=["energy", "cvar"], default="energy")
    parser.add_argument("--cvar-alpha", type=float, default=0.1)
    parser.add_argument("--backend", choices=["aer", "statevector", "fastsim"], default="aer")
    parser.add_argument("--root", type=Path, default=Path("runs"))
    args = parser.parse_args()

    problem = problem_from_scene(load_scene(args.scene))
    A, B = default_penalties(problem.v)
    qubo = build_qubo(problem.v, problem.dw, problem.dW, A, B)
    dp = dp_solve(problem.v, problem.dw, problem.dW)

    results = run_qaoa_sweep(
        qubo,
        args.p,
        args.seed,
        n_starts=args.n_starts,
        maxiter=args.maxiter,
        shots=args.shots,
        objective=args.objective,
        cvar_alpha=args.cvar_alpha,
        backend=args.backend,
    )

    name = args.scene.stem
    print(
        f"scene {name}: {qubo.n} qubits, A = {A:.3f}, B = {B:.3f}, DP {dp.choice} V = {dp.value:.4f}"
    )
    print(f"{'p':>2} {'<H>':>9} {'P_feas':>7} {'P_dp':>7} {'evals':>6} {'time_s':>7}  run")
    for result in results:
        decoded = {bitstring: bits_from_bitstring(bitstring, qubo.n) for bitstring in result.counts}
        feasible = sum(
            count
            for bitstring, count in result.counts.items()
            if is_feasible(decoded[bitstring], qubo, problem.dw)
        )
        optimal = sum(
            count
            for bitstring, count in result.counts.items()
            if is_feasible(decoded[bitstring], qubo, problem.dw)
            and decode(decoded[bitstring], qubo) == dp.choice
        )
        extra = {
            "problem": {"v": problem.v, "dw": problem.dw, "dW": problem.dW},
            "penalties": {"A": A, "B": B},
            "dp": {"choice": dp.choice, "value": dp.value, "used": dp.used},
        }
        path = save_qaoa_run(result, name, root=args.root, extra=extra)
        print(
            f"{result.p:2d} {result.best_expectation:9.3f} {feasible / args.shots:7.3f} "
            f"{optimal / args.shots:7.3f} {result.n_evals:6d} {result.wall_time_s:7.1f}  {path}"
        )


if __name__ == "__main__":
    main()
