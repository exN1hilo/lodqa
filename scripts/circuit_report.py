"""Print the depth and two-qubit gate count of the 19-qubit QAOA circuit for p = 1..4."""

import argparse
from pathlib import Path

from lodq.ising import qubo_to_ising
from lodq.metrics import problem_from_scene
from lodq.qaoa import build_qaoa_circuit, circuit_stats, normalize_operator, transpile_for_aer
from lodq.qubo import build_qubo, default_penalties
from lodq.scene import load_scene


def main() -> None:
    parser = argparse.ArgumentParser(description="QAOA circuit stats after transpiling for Aer")
    parser.add_argument("--scene", type=Path, default=Path("scenes/default.yaml"))
    parser.add_argument("--p", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    problem = problem_from_scene(load_scene(args.scene))
    A, B = default_penalties(problem.v)
    op, _ = qubo_to_ising(build_qubo(problem.v, problem.dw, problem.dW, A, B))
    cost_op, _ = normalize_operator(op)

    print(f"scene {args.scene.stem}: dW = {problem.dW}, qubits = {problem.n_qubits}")
    print(f"{'p':>2} {'depth':>6} {'depth_2q':>8} {'n_2q':>5}  count_ops")
    for p in args.p:
        stats = circuit_stats(transpile_for_aer(build_qaoa_circuit(cost_op, p), args.seed))
        print(
            f"{p:2d} {stats['depth']:6d} {stats['depth_2q']:8d} {stats['n_2q']:5d}  "
            f"{stats['count_ops']}"
        )


if __name__ == "__main__":
    main()
