"""Search for scene presets where greedy LOD selection leaves unusable budget."""

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from lodq.classical import (
    Solution,
    approximation_ratio,
    dp_solve,
    enumerate_feasible,
    greedy_solve,
)
from lodq.metrics import Problem, problem_from_scene
from lodq.scene import Scene, camera_path, in_view, load_scene, random_scene, save_scene

PATH_FRAMES = 41
PATH_HALF_LENGTH = 10.0


@dataclass(frozen=True, slots=True)
class TrapCandidate:
    scene: Scene
    problem: Problem
    dp: Solution
    greedy: Solution
    ratio: float
    optimality_gap: float
    minimum_unclaimed_upgrade: int

    @property
    def rank(self) -> tuple[float, float, float, int, str]:
        return (
            abs(self.ratio - 0.85),
            -self.optimality_gap,
            -self.dp.used / self.problem.dW,
            self.greedy.used,
            self.scene.name,
        )


def _path_frames(scene: Scene, n_frames: int = PATH_FRAMES) -> list[Scene]:
    """Camera pass along z through the scene camera, PATH_HALF_LENGTH to each side."""
    x, y, z = scene.camera.position
    return camera_path(
        scene,
        n_frames=n_frames,
        start=(x, y, z - PATH_HALF_LENGTH),
        end=(x, y, z + PATH_HALF_LENGTH),
    )


def _visible_on_path(scene: Scene) -> bool:
    return all(bool(in_view(frame).all()) for frame in _path_frames(scene))


def _greedy_recovers_on_path(scene: Scene) -> bool:
    """True if Greedy is optimal on at least one frame, so the path shows the gap opening."""
    for frame in _path_frames(scene):
        problem = problem_from_scene(frame)
        optimum = dp_solve(problem.v, problem.dw, problem.dW)
        greedy = greedy_solve(problem.v, problem.dw, problem.dW)
        if approximation_ratio(greedy, optimum) >= 1.0 - 1e-9:
            return True
    return False


def _candidate_scene(base: Scene, seed: int, attempt: int) -> Scene:
    sample_seed = seed + 17 * attempt
    rng = np.random.default_rng(sample_seed)
    # The path ends PATH_HALF_LENGTH ahead, so keep its last frame short of the target.
    camera_position = (
        float(rng.uniform(-10.0, 10.0)),
        base.camera.position[1],
        float(rng.uniform(-10.0, base.camera.target[2] - PATH_HALF_LENGTH - 2.0)),
    )
    camera = replace(base.camera, position=camera_position)
    # Place objects from the closest camera on the path, where the view is narrowest.
    path_end = (camera_position[0], camera_position[1], camera_position[2] + PATH_HALF_LENGTH)
    generated = random_scene(
        sample_seed,
        n_objects=base.n_objects,
        n_lods=base.n_lods,
        dW_range=(8, 15),
        block_size=base.block_size,
        camera=replace(camera, position=path_end),
    )
    objects = tuple(
        replace(
            original,
            position=sampled.position,
            radius=original.radius * float(rng.uniform(0.5, 1.5)),
            importance=original.importance * float(rng.uniform(0.5, 1.5)),
            tri_counts=sampled.tri_counts,
        )
        for original, sampled in zip(base.objects, generated.objects, strict=True)
    )
    w_base = sum(obj.tri_counts[-1] for obj in objects)
    generated_dw = problem_from_scene(generated).dW
    budget = w_base + generated_dw * base.block_size
    return replace(
        base,
        name=f"greedy_trap_candidate_{attempt:04d}",
        objects=objects,
        camera=camera,
        budget_W=budget,
    )


def _evaluate_candidate(scene: Scene) -> TrapCandidate | None:
    problem = problem_from_scene(scene)
    if problem.n_qubits > 19 or problem.dW <= 0:
        return None
    if not _visible_on_path(scene):
        return None

    optimum = dp_solve(problem.v, problem.dw, problem.dW)
    greedy = greedy_solve(problem.v, problem.dw, problem.dW)
    ratio = approximation_ratio(greedy, optimum)
    if not (
        0.82 <= ratio <= 0.88
        and optimum.used >= 0.90 * problem.dW
        and greedy.used <= 0.85 * problem.dW
    ):
        return None

    feasible = enumerate_feasible(problem.v, problem.dw, problem.dW)
    if len(feasible) < 2:
        return None
    gap = feasible[0].value - feasible[1].value
    if gap < 0.03:
        return None

    remaining = problem.dW - greedy.used
    untouched = [
        i for i, choice in enumerate(greedy.choice) if choice == problem.M - 1
    ]
    upgrade_costs = [
        int(problem.dw[i, lod] - problem.dw[i, greedy.choice[i]])
        for i in untouched
        for lod in range(greedy.choice[i])
    ]
    if not upgrade_costs or min(upgrade_costs) <= remaining:
        return None
    if not _greedy_recovers_on_path(scene):
        return None

    return TrapCandidate(
        scene=scene,
        problem=problem,
        dp=optimum,
        greedy=greedy,
        ratio=ratio,
        optimality_gap=gap,
        minimum_unclaimed_upgrade=min(upgrade_costs),
    )


def search_traps(base: Scene, seed: int, attempts: int = 50000) -> list[TrapCandidate]:
    """Return up to three distinct candidates satisfying all trap criteria."""
    if attempts < 1:
        raise ValueError("attempts must be positive")
    if base.n_objects != 5 or base.n_lods != 3:
        raise ValueError("trap search requires exactly five objects and three LODs")

    best: list[TrapCandidate] = []
    for attempt in range(attempts):
        candidate = _evaluate_candidate(_candidate_scene(base, seed, attempt))
        if candidate is None:
            continue
        if any(item.scene == candidate.scene for item in best):
            continue
        best.append(candidate)
        best.sort(key=lambda item: item.rank)
        del best[3:]
    return best


def camera_path_report(candidate: TrapCandidate, n_frames: int = PATH_FRAMES) -> dict[str, Any]:
    """Evaluate the greedy approximation ratio along a camera pass through the trap."""
    frames = _path_frames(candidate.scene, n_frames)
    frame_results = []
    for index, frame in enumerate(frames):
        problem = problem_from_scene(frame)
        optimum = dp_solve(problem.v, problem.dw, problem.dW)
        greedy = greedy_solve(problem.v, problem.dw, problem.dW)
        frame_results.append(
            {
                "frame": index,
                "camera_position": list(frame.camera.position),
                "ar_greedy": approximation_ratio(greedy, optimum),
                "dp_value": optimum.value,
                "greedy_value": greedy.value,
                "dp_used": optimum.used,
                "greedy_used": greedy.used,
                "budget": problem.dW,
            }
        )
    return {"scene": candidate.scene.name, "frames": frame_results}


def _save_results(
    candidates: list[TrapCandidate],
    output_dir: Path,
    n_frames: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, candidate in enumerate(candidates, start=1):
        save_scene(
            replace(candidate.scene, name=f"greedy_trap_{index}"),
            output_dir / f"greedy_trap_{index}.yaml",
        )
    if candidates:
        path_report = camera_path_report(
            replace(candidates[0], scene=replace(candidates[0].scene, name="greedy_trap_1")),
            n_frames=n_frames,
        )
        (output_dir / "greedy_trap_path.json").write_text(
            json.dumps(path_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Найти сцены-ловушки для Greedy")
    parser.add_argument("--scene", type=Path, default=Path("scenes/default.yaml"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attempts", type=int, default=50000)
    parser.add_argument("--out", type=Path, default=Path("scenes"))
    parser.add_argument("--frames", type=int, default=PATH_FRAMES)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.frames < 2:
        parser.error("--frames must be at least 2")

    candidates = search_traps(load_scene(args.scene), args.seed, args.attempts)
    _save_results(candidates, args.out, args.frames)
    print(f"{'scene':<18} {'AR':>6} {'DP/budget':>12} {'Greedy/budget':>15} {'gap':>8}")
    for index, candidate in enumerate(candidates, start=1):
        print(
            f"greedy_trap_{index:<7} {candidate.ratio:6.3f} "
            f"{candidate.dp.used:4d}/{candidate.problem.dW:<7d} "
            f"{candidate.greedy.used:4d}/{candidate.problem.dW:<10d} "
            f"{candidate.optimality_gap:8.3f}"
        )
    if len(candidates) < 3:
        raise RuntimeError(
            f"Found only {len(candidates)} of 3 scenes in {args.attempts} attempts; "
            "increase --attempts or change --seed."
        )


if __name__ == "__main__":
    main()
