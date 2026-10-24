import argparse
from pathlib import Path

from lodq.classical import dp_solve
from lodq.metrics import problem_from_scene
from lodq.qubo import build_qubo, cache_energies, calibrate_penalties, energies_cache_path
from lodq.scene import load_scene
from lodq.storage import save_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Откалибровать штрафы для всех YAML-сцен")
    parser.add_argument("--scenes-dir", type=Path, default=Path("scenes"))
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    args = parser.parse_args()

    paths = sorted(args.scenes_dir.glob("*.yaml"))
    if not paths:
        parser.error(f"В {args.scenes_dir} не найдено YAML-сцен")

    rows = []
    for path in paths:
        name = path.stem
        scene = load_scene(path)
        problem = problem_from_scene(scene)
        solution = dp_solve(problem.v, problem.dw, problem.dW)
        A, B, report = calibrate_penalties(problem.v, problem.dw, problem.dW, solution)
        qubo = build_qubo(problem.v, problem.dw, problem.dW, A, B)
        cache_path = args.cache_dir / energies_cache_path(name, qubo).name
        cache_energies(qubo, cache_path)
        save_run(
            {
                "scene": name,
                "A": A,
                "B": B,
                "gap_to_infeasible": report.gap_to_infeasible,
                "gap_to_second_feasible": report.gap_to_second_feasible,
                "energies_path": str(cache_path),
            },
            tag=f"calib__{name}",
            root=args.runs_root,
        )
        rows.append((name, A, B, report.gap_to_infeasible, report.gap_to_second_feasible))

    print(
        f"{'scene':<20} {'A':>12} {'B':>12} {'gap_to_infeasible':>20} {'gap_to_second_feasible':>24}"
    )
    for name, A, B, gap_infeasible, gap_second in rows:
        print(f"{name:<20} {A:12.6g} {B:12.6g} {gap_infeasible:20.6g} {gap_second:24.6g}")


if __name__ == "__main__":
    main()
