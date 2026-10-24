"""Save a suite of random scenes and print dW and the qubit count."""

import argparse
from pathlib import Path

from lodq.metrics import problem_from_scene
from lodq.scene import save_scene, scene_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate random scene presets")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("scenes"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    scenes = scene_suite(range(args.count))
    print(f"{'name':<16} {'dW':>4} {'n_qubits':>8}")
    for seed, scene in zip(range(args.count), scenes, strict=True):
        problem = problem_from_scene(scene)
        save_scene(scene, args.out / f"random_{seed:03d}.yaml")
        print(f"{scene.name:<16} {problem.dW:4d} {problem.n_qubits:8d}")


if __name__ == "__main__":
    main()
