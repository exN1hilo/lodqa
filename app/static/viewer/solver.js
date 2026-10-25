// Browser port of lodq.metrics (utility, delta weights) and lodq.classical (DP, Greedy).
// The 3D explorer runs these on every camera move, so the LOD choice follows the camera
// in real time. tests/test_viewer_js.py checks them against the Python solvers.

export const ALPHA = 1.5;
export const QUALITY = [1.0, 0.65, 0.25];
const TIE_TOL = 1e-12;

export function distance(a, b) {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  const dz = a[2] - b[2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

// v[i][j] = S_i * q[j] / d_i^alpha with S_i = 4*pi*R_i^2 * importance_i, max v = 1.
export function utilityMatrix(objects, camera, alpha = ALPHA, q = QUALITY) {
  const raw = objects.map((obj) => {
    const d = Math.max(distance(obj.position, camera), 1e-6);
    const area = 4.0 * Math.PI * (obj.radius * obj.radius) * obj.importance;
    const scale = area / Math.pow(d, alpha);
    return q.map((quality) => scale * quality);
  });
  const top = Math.max(...raw.flat());
  return raw.map((row) => row.map((value) => value / top));
}

// dw[i][j] = ceil((w[i][j] - w[i][M-1]) / block), dW = floor((budget - W_base) / block).
export function deltaWeights(triCounts, budget, blockSize) {
  const dw = triCounts.map((row) => {
    const base = row[row.length - 1];
    return row.map((w, j) => (j === row.length - 1 ? 0 : Math.floor((w - base + blockSize - 1) / blockSize)));
  });
  const wBase = triCounts.reduce((sum, row) => sum + row[row.length - 1], 0);
  return { dw, dW: Math.floor((budget - wBase) / blockSize), wBase };
}

export function evaluate(choice, v, dw, triCounts) {
  let value = 0;
  let used = 0;
  let triangles = 0;
  choice.forEach((j, i) => {
    value += v[i][j];
    used += dw[i][j];
    triangles += triCounts ? triCounts[i][j] : 0;
  });
  return { choice, value, used, triangles };
}

function lexLess(a, b) {
  for (let k = 0; k < a.length; k += 1) {
    if (a[k] !== b[k]) return a[k] < b[k];
  }
  return false;
}

// Exact optimum with the tie rule of dp_solve: highest value (within 1e-12), then smaller
// used, then the lexicographically smaller choice. Enumerates all M^N choices (243 here).
export function dpSolve(v, dw, dW) {
  const n = v.length;
  const m = v[0].length;
  if (dW < 0) return null;
  const total = m ** n;
  if (total > 2_000_000) throw new Error(`too many choices to enumerate: ${total}`);
  let best = null;
  const choice = new Array(n).fill(0);
  for (let code = 0; code < total; code += 1) {
    let rest = code;
    for (let i = n - 1; i >= 0; i -= 1) {
      choice[i] = rest % m;
      rest = Math.floor(rest / m);
    }
    const candidate = evaluate(choice, v, dw);
    if (candidate.used > dW) continue;
    if (
      best === null ||
      candidate.value > best.value + TIE_TOL ||
      (Math.abs(candidate.value - best.value) <= TIE_TOL &&
        (candidate.used < best.used || (candidate.used === best.used && lexLess(choice, best.choice))))
    ) {
      best = { ...candidate, choice: [...choice] };
    }
  }
  return best;
}

function tupleGreater(a, b) {
  for (let k = 0; k < a.length; k += 1) {
    if (a[k] !== b[k]) return a[k] > b[k];
  }
  return false;
}

// greedy_solve: upgrade from the lowest LOD by marginal utility per budget unit. Free
// upgrades go first; ties prefer the larger gain, then the smaller i, then the smaller j.
export function greedySolve(v, dw, dW) {
  const n = v.length;
  const m = v[0].length;
  if (dW < 0) return null;
  const current = new Array(n).fill(m - 1);
  let used = 0;
  for (;;) {
    let free = null;
    let paid = null;
    for (let i = 0; i < n; i += 1) {
      for (let j = 0; j < current[i]; j += 1) {
        const ddv = v[i][j] - v[i][current[i]];
        const ddw = dw[i][j] - dw[i][current[i]];
        if (ddv <= 0 || used + ddw > dW) continue;
        if (ddw <= 0) {
          const key = [ddv, -i, -j];
          if (free === null || tupleGreater(key, free.key)) free = { key, i, j, ddw };
        } else {
          const key = [ddv / ddw, ddv, -i, -j];
          if (paid === null || tupleGreater(key, paid.key)) paid = { key, i, j, ddw };
        }
      }
    }
    const step = free ?? paid;
    if (step === null) break;
    current[step.i] = step.j;
    used += step.ddw;
  }
  return evaluate(current, v, dw);
}

// Everything the explorer needs for one camera position and budget.
export function solveView(objects, camera, budget, blockSize, alpha = ALPHA, q = QUALITY) {
  const triCounts = objects.map((obj) => obj.triCounts);
  const v = utilityMatrix(objects, camera, alpha, q);
  const { dw, dW, wBase } = deltaWeights(triCounts, budget, blockSize);
  const distances = objects.map((obj) => distance(obj.position, camera));
  if (dW < 0) return { v, dw, dW, wBase, distances, feasible: false, dp: null, greedy: null };
  const withTriangles = (solution) => evaluate(solution.choice, v, dw, triCounts);
  return {
    v,
    dw,
    dW,
    wBase,
    distances,
    feasible: true,
    dp: withTriangles(dpSolve(v, dw, dW)),
    greedy: withTriangles(greedySolve(v, dw, dW)),
  };
}
