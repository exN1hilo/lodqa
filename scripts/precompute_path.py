"""Precompute DP, Greedy and QAOA for every camera frame of the greedy trap path."""

import argparse
import json
import time
from pathlib import Path

from lodq.demo import load_path, precompute_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute the camera path for the demo app")
    parser.add_argument("--path", type=Path, default=Path("scenes/greedy_trap_path.json"))
    parser.add_argument("--scene", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("cache/path"))
    parser.add_argument("--limit", type=int, default=None, help="only the first N frames")
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-starts", type=int, default=3)
    parser.add_argument("--maxiter", type=int, default=60, help="first frame")
    parser.add_argument("--warm-maxiter", type=int, default=30, help="warm-started frames")
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--backend", choices=["fastsim", "aer", "statevector"], default="fastsim")
    parser.add_argument("--exact", action="store_true", help="also store exact metrics")
    args = parser.parse_args()

    scene_name, cameras = load_path(args.path)
    scene_path = args.scene or Path("scenes") / f"{scene_name}.yaml"
    cameras = cameras[: args.limit]

    started = time.perf_counter()
    paths = precompute_path(
        scene_path,
        cameras,
        args.out,
        p=args.p,
        seed=args.seed,
        n_starts=args.n_starts,
        maxiter=args.maxiter,
        warm_maxiter=args.warm_maxiter,
        shots=args.shots,
        backend=args.backend,
        exact=args.exact,
    )

    print(f"{'frame':>5} {'AR_greedy':>9} {'AR_qaoa':>8} {'P_feas':>7} {'P_opt':>7} {'time_s':>7}")
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        metrics = record["metrics"]
        print(
            f"{record['frame']:5d} {record['ar_greedy']:9.3f} {metrics['ar_top1']:8.3f} "
            f"{metrics['p_feas']:7.3f} {metrics['p_opt']:7.3f} {record['timings']['frame_s']:7.1f}"
        )
    print(f"{len(paths)} frames in {time.perf_counter() - started:.1f} s -> {args.out}")


if __name__ == "__main__":
    main()
