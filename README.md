# lod-qaoa

Choosing a level of detail (LOD) for each object in a 3D scene under a hard per-frame
triangle budget, written as a QUBO and solved with QAOA (Qiskit), then compared honestly
with exact dynamic programming, a greedy baseline and random sampling.

![DP, Greedy and QAOA choices rendered side by side](figures/compare_render.png)

## Results

<!-- RESULTS:START -->

_No results yet. Generate the table with `uv run python scripts/results_table.py --write-readme`._

<!-- RESULTS:END -->

AR top-1 is the value of the most frequent feasible sample divided by the DP optimum.
P_feas is the fraction of shots that are one-hot and within budget. Budget use is the
budget used by the best feasible sample divided by the available budget. P_opt gain is
the probability of sampling the DP optimum divided by its probability under uniform
sampling.

## Method

**Problem.** A scene has N = 5 objects, each with M = 3 LODs: j = 0 is the most detailed
and j = 2 the coarsest. Exactly one LOD is picked per object, and the total triangle
count must fit the frame budget while maximising visual utility. This is a
multiple-choice knapsack problem (MCKP).

**Utility.** For object i at LOD j,
v<sub>ij</sub> = S<sub>i</sub> · q<sub>j</sub> / d<sub>i</sub><sup>α</sup>, where
S<sub>i</sub> = 4πR<sub>i</sub><sup>2</sup> · importance<sub>i</sub>,
d<sub>i</sub> is the distance to the camera, α = 1.5 and q = (1.0, 0.65, 0.25).
v is normalised so that max v = 1.

**Delta weights and budget quantisation.** Every object costs at least its coarsest LOD,
so that cost is moved into a fixed base W<sub>base</sub> = Σ<sub>i</sub> w<sub>i,M−1</sub>.
The remaining weights and budget are counted in blocks of 5000 triangles:

- dw<sub>ij</sub> = ⌈(w<sub>ij</sub> − w<sub>i,M−1</sub>) / 5000⌉, so dw<sub>i,M−1</sub> = 0
- dW = ⌊(W<sub>max</sub> − W<sub>base</sub>) / 5000⌋, typically 8 to 15

Rounding the weights up and the budget down means that every choice the model accepts
also fits the real triangle budget.

**QUBO.** Binary x<sub>ij</sub> = 1 means object i uses LOD j. The inequality is turned
into an equality with K = ⌊log<sub>2</sub> dW⌋ + 1 slack bits s<sub>k</sub> of weight 2<sup>k</sup>.
The energy to minimise is

$$
E(x, s) = -\sum_{i,j} v_{ij} x_{ij}
+ A \sum_i \Big(1 - \sum_j x_{ij}\Big)^2
+ B \Big(\sum_{i,j} dw_{ij} x_{ij} + \sum_k 2^k s_k - dW\Big)^2
$$

**Qubits.** With dW ≤ 15 this gives N·M = 15 decision qubits plus K = 4 slack qubits,
19 qubits in total.

**Penalties.** A = 1.5 · max v. Going over the budget by one block costs only B, so B
must be larger than the biggest gain from a single upgrade,
B > max<sub>ij</sub> (v<sub>ij</sub> − v<sub>i,M−1</sub>). We use 1.2 times that value.
The common rule of thumb B ≈ max(Δv/Δw) is not safe here.

**Verification.** For every scene, all 2<sup>19</sup> states are enumerated
(`lodq.qubo.verify_qubo`). The penalties are accepted only if the ground state is
unique, feasible and equal to the DP solution, with an energy margin to every
infeasible state. The QUBO → Ising mapping is checked the same way against a statevector.

**QAOA.** The standard X mixer, and an XY mixer that keeps the one-hot constraint
(started from W states, with A = 0). Angles are optimised with multistart COBYLA using
linear-ramp, random and warm starts from depth p − 1. Experiments use an exact numpy
statevector (`fastsim`), and the final counts are sampled with Qiskit Aer.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is installed by uv.

```bash
uv sync && make all QUICK=1   # install, test, build assets, quick experiments, figures, renders
make demo                     # interactive Streamlit demo
```

`make all` without `QUICK=1` runs the full grid from `configs/main.yaml`.

## Repository layout

| Module | Purpose |
|---|---|
| `src/lodq/scene.py` | Scene and camera data model, YAML I/O, random scenes and camera paths |
| `src/lodq/metrics.py` | Utility matrix, delta weights, budget quantisation and qubit count |
| `src/lodq/meshes.py` | Builds the LOD meshes in `assets/lods/` by mesh simplification |
| `src/lodq/classical.py` | Exact DP, brute force, greedy and random baselines |
| `src/lodq/qubo.py` | QUBO construction, encoding and decoding, penalty rules, full 2^n verification |
| `src/lodq/ising.py` | QUBO → Ising (`SparsePauliOp`) mapping and its diagonal |
| `src/lodq/qaoa.py` | QAOA circuits (X and XY mixers), angle initialisation, optimisation and sampling |
| `src/lodq/fastsim.py` | Exact numpy QAOA statevector, fast optimiser and the p = 1 landscape |
| `src/lodq/evaluate.py` | Sample and distribution metrics (AR, P_feas, P_opt gain) and the uniform baseline |
| `src/lodq/experiments.py` | Command-line runner for the experiment grid and `runs/summary.csv` |
| `src/lodq/storage.py` | Saving and loading run records under `runs/` |
| `src/lodq/config.py` | Experiment configuration schema (pydantic) |
| `src/lodq/plots.py` | Figures rebuilt from saved runs |
| `src/lodq/render.py` | 3D renders of the chosen LODs with PyVista |
| `src/lodq/demo.py` | Data and figures for the Streamlit demo in `app/` |

## Limitations

- **DP is better at this size.** For 5 objects × 3 LODs, dynamic programming solves the
  problem exactly in under a millisecond. QAOA is neither faster nor more accurate here.
- **No quantum advantage is claimed.** The project shows a correct formulation and a
  pipeline that is verified end to end: exhaustive QUBO check, cross-checked simulators
  and honest baselines.
- **Noiseless simulation only.** All results come from a noise-free simulator. Real
  hardware noise and the depth of the transpiled circuit would lower the quality.

## Team

- Volodymyr Cherevko
- Ruslan Murha
- Artem Bondarenko
- kolodoschi-two
- _TODO: name of the contributor committing as iceice_
