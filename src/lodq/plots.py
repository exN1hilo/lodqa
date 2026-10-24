"""Figures rebuilt from saved runs, without launching QAOA again."""

import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lodq.classical import Solution
from lodq.evaluate import top_k
from lodq.qubo import build_qubo
from lodq.storage import load_runs, runs_to_dataframe

COLORS = {
    "dp": "#3a3a3a",
    "greedy": "#e07b00",
    "random": "#b0b0b0",
    "qaoa_x": "#1f6fd1",
    "qaoa_xy": "#7b3fc4",
}


def apply_style() -> None:
    """12 pt type, open top and right spines, light y-grid, 12.8 by 7.2 inch figures."""
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": "#e0e0e0",
            "grid.linewidth": 0.8,
            "figure.figsize": (12.8, 7.2),
            "axes.axisbelow": True,
            "legend.frameon": False,
        }
    )


def save_figure(fig: plt.Figure, name: str, out_dir: str | Path = "figures") -> list[Path]:
    """Write PNG at 300 dpi and SVG."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    png = directory / f"{name}.png"
    svg = directory / f"{name}.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    return [png, svg]


def label_line_end(ax: plt.Axes, x: float, y: float, text: str, color: str) -> None:
    """Place a label at the end of a line instead of using a legend."""
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(8, 0),
        textcoords="offset points",
        color=color,
        va="center",
        ha="left",
        clip_on=False,
    )


def _subset(df: pd.DataFrame, scene: str) -> pd.DataFrame:
    if "scene" not in df.columns:
        return df.copy()
    return df.loc[df["scene"] == scene].copy()


def _metric_frame(df: pd.DataFrame, scene: str, metric: str) -> pd.DataFrame:
    """Rows of p, mixer, mean, std for one metric, from raw runs or summary.csv."""
    data = _subset(df, scene)
    mixer_col = "qaoa.mixer" if "qaoa.mixer" in data.columns else "mixer"
    p_col = "qaoa.p" if "qaoa.p" in data.columns else "p"
    mean_col = f"metrics.{metric}_mean"
    raw_col = f"metrics.{metric}"
    if mean_col in data.columns:
        std_col = f"metrics.{metric}_std"
        frame = data[[p_col, mixer_col, mean_col]].rename(
            columns={p_col: "p", mixer_col: "mixer", mean_col: "mean"}
        )
        frame["std"] = data[std_col].to_numpy() if std_col in data.columns else 0.0
    else:
        frame = (
            data.groupby([p_col, mixer_col], dropna=False)[raw_col]
            .agg(mean="mean", std="std")
            .reset_index()
            .rename(columns={p_col: "p", mixer_col: "mixer"})
        )
    frame["std"] = frame["std"].fillna(0.0)
    return frame.sort_values(["mixer", "p"])


def _level(df: pd.DataFrame, scene: str, column: str) -> float | None:
    data = _subset(df, scene)
    if column not in data.columns or data[column].dropna().empty:
        return None
    return float(data[column].median())


def _greedy_ar(df: pd.DataFrame, scene: str) -> float | None:
    data = _subset(df, scene)
    if {"greedy.value", "dp.value"} <= set(data.columns):
        ratio = data["greedy.value"] / data["dp.value"]
        if ratio.dropna().empty:
            return None
        return float(ratio.median())
    return _level(df, scene, "greedy.ar")


def _new_ax(title: str) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.grid(True, axis="y", color="#e0e0e0")
    return fig, ax


def _mixer_color(mixer: str) -> str:
    return COLORS["qaoa_xy"] if mixer == "xy" else COLORS["qaoa_x"]


def _mixer_name(mixer: str) -> str:
    return "XY" if mixer == "xy" else "X"


def fig_ar_vs_p(df: pd.DataFrame, scene: str = "greedy_trap_1") -> plt.Figure:
    """Approximation ratio versus depth, with DP, greedy, and random references."""
    stats = _metric_frame(df, scene, "ar_top1")
    best = stats.loc[stats["mean"].idxmax()]
    title = (
        f"QAOA-{_mixer_name(str(best['mixer']))} at p={int(best['p'])} "
        f"reaches an approximation ratio of {best['mean']:.2f}."
    )
    fig, ax = _new_ax(title)
    for mixer, group in stats.groupby("mixer"):
        color = _mixer_color(str(mixer))
        ax.errorbar(
            group["p"],
            group["mean"],
            yerr=group["std"],
            color=color,
            marker="o",
            capsize=3,
            linewidth=2,
        )
        last = group.iloc[-1]
        label_line_end(
            ax, float(last["p"]), float(last["mean"]), f"QAOA {_mixer_name(str(mixer))}", color
        )
    right = float(stats["p"].max())
    ax.axhline(1.0, color=COLORS["dp"], linewidth=1.5)
    label_line_end(ax, right, 1.0, "DP", COLORS["dp"])
    greedy = _greedy_ar(df, scene)
    if greedy is not None:
        ax.axhline(greedy, color=COLORS["greedy"], linewidth=1.5)
        label_line_end(ax, right, greedy, "Greedy", COLORS["greedy"])
    random_ar = _level(df, scene, "random.ar_mean_feasible")
    if random_ar is not None:
        ax.axhline(random_ar, color=COLORS["random"], linewidth=1.5)
        label_line_end(ax, right, random_ar, "Random", COLORS["random"])
    ax.set_xlabel("p")
    ax.set_ylabel("approximation ratio")
    ax.set_xticks(sorted(stats["p"].unique()))
    save_figure(fig, "ar_vs_p")
    return fig


def fig_feasibility(df: pd.DataFrame, scene: str = "greedy_trap_1") -> plt.Figure:
    """Feasible and one-hot shot fractions versus depth."""
    feasible = _metric_frame(df, scene, "p_feas")
    onehot = _metric_frame(df, scene, "p_onehot")
    best = feasible.loc[feasible["mean"].idxmax()]
    title = (
        f"At p={int(best['p'])}, {100 * best['mean']:.0f}% of "
        f"QAOA-{_mixer_name(str(best['mixer']))} shots are feasible."
    )
    fig, ax = _new_ax(title)
    for mixer, group in feasible.groupby("mixer"):
        color = _mixer_color(str(mixer))
        ax.plot(group["p"], group["mean"], color=color, linewidth=2)
        last = group.iloc[-1]
        label_line_end(
            ax, float(last["p"]), float(last["mean"]), f"{_mixer_name(str(mixer))} feasible", color
        )
    for mixer, group in onehot.groupby("mixer"):
        color = _mixer_color(str(mixer))
        ax.plot(group["p"], group["mean"], color=color, linewidth=2, linestyle="--")
        last = group.iloc[-1]
        label_line_end(
            ax, float(last["p"]), float(last["mean"]), f"{_mixer_name(str(mixer))} one-hot", color
        )
    ax.set_xlabel("p")
    ax.set_ylabel("shot fraction")
    ax.set_xticks(sorted(feasible["p"].unique()))
    ax.set_ylim(0, 1.05)
    save_figure(fig, "feasibility")
    return fig


def fig_amplification(df: pd.DataFrame, scene: str = "greedy_trap_1") -> plt.Figure:
    """How many times more often QAOA hits the optimum than a uniform draw."""
    stats = _metric_frame(df, scene, "amplification")
    best = stats.loc[stats["mean"].idxmax()]
    title = (
        f"At p={int(best['p'])}, QAOA-{_mixer_name(str(best['mixer']))} finds the optimum "
        f"{best['mean']:.0f} times more often than chance."
    )
    fig, ax = _new_ax(title)
    for mixer, group in stats.groupby("mixer"):
        color = _mixer_color(str(mixer))
        ax.plot(group["p"], group["mean"], color=color, marker="o", linewidth=2)
        last = group.iloc[-1]
        label_line_end(
            ax, float(last["p"]), float(last["mean"]), f"QAOA {_mixer_name(str(mixer))}", color
        )
    right = float(stats["p"].max())
    ax.axhline(1.0, color=COLORS["random"], linewidth=1.5)
    label_line_end(ax, right, 1.0, "Random", COLORS["random"])
    ax.set_yscale("log")
    ax.set_xlabel("p")
    ax.set_ylabel("amplification")
    ax.set_xticks(sorted(stats["p"].unique()))
    save_figure(fig, "amplification")
    return fig


def _qaoa(run: dict[str, Any]) -> dict[str, Any]:
    qaoa = run.get("qaoa")
    if isinstance(qaoa, dict):
        return qaoa
    return {
        "p": run.get("qaoa.p", run.get("p")),
        "seed": run.get("qaoa.seed", run.get("seed")),
        "mixer": run.get("qaoa.mixer", run.get("mixer", "x")),
        "counts": run.get("qaoa.counts", run.get("counts", {})),
        "energy_history": run.get("qaoa.energy_history", run.get("energy_history", [])),
        "gammas": run.get("qaoa.gammas", run.get("gammas", [])),
        "betas": run.get("qaoa.betas", run.get("betas", [])),
    }


def _problem_arrays(run: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, int, float, float]:
    problem = run["problem"]
    penalties = run["penalties"]
    return (
        np.asarray(problem["v"], dtype=np.float64),
        np.asarray(problem["dw"]),
        int(problem["dW"]),
        float(penalties["A"]),
        float(penalties["B"]),
    )


def fig_top10(run: dict[str, Any]) -> plt.Figure:
    """Ten most common bitstrings for one saved run."""
    utility, weights, budget, penalty_a, penalty_b = _problem_arrays(run)
    qubo = build_qubo(utility, weights, budget, penalty_a, penalty_b)
    dp = run["dp"]
    optimum = Solution(
        tuple(dp["choice"]), float(dp["value"]), int(dp["used"]), str(dp.get("solver", "dp"))
    )
    rows = list(reversed(top_k(_qaoa(run)["counts"], qubo, utility, weights, optimum, k=10)))
    leader = rows[-1]
    title = f"The most common sample is {leader['lod_code']} with probability {float(leader['prob']):.2f}."
    fig, ax = _new_ax(title)
    labels = [str(row["lod_code"]) for row in rows]
    for index, row in enumerate(rows):
        if row["is_optimal"]:
            color, hatch = COLORS["qaoa_x"], None
        elif row["feasible"]:
            color, hatch = COLORS["random"], None
        else:
            color, hatch = "#e8e8e8", "//"
        ax.barh(index, float(row["prob"]), color=color, hatch=hatch, edgecolor="#666666")
    ax.set_yticks(range(len(rows)), labels)
    ax.set_xlabel("probability")
    save_figure(fig, "top10")
    return fig


def fig_convergence(runs: list[dict[str, Any]]) -> plt.Figure:
    """COBYLA energy history for p = 1..4 of one scene and one seed."""
    ordered = sorted(runs, key=lambda run: int(_qaoa(run)["p"]))
    deepest = ordered[-1]
    history = [float(value) for value in _qaoa(deepest)["energy_history"]]
    title = (
        f"The p={int(_qaoa(deepest)['p'])} run reaches energy {history[-1]:.3g} "
        f"after {len(history)} evaluations."
    )
    fig, ax = _new_ax(title)
    for run in ordered:
        qaoa = _qaoa(run)
        values = [float(value) for value in qaoa["energy_history"]]
        color = _mixer_color(str(qaoa.get("mixer", "x")))
        ax.plot(range(1, len(values) + 1), values, color=color, linewidth=2)
        label_line_end(ax, len(values), values[-1], f"p={int(qaoa['p'])}", color)
    ax.set_xlabel("evaluation")
    ax.set_ylabel("energy")
    save_figure(fig, "convergence")
    return fig


def fig_landscape_p1(landscape_npz: str | Path, run_p1: dict[str, Any]) -> plt.Figure:
    """Cost landscape over gamma and beta at p = 1, with the COBYLA point marked."""
    archive = np.load(landscape_npz)
    gamma = np.asarray(
        archive["gamma"] if "gamma" in archive else archive["gammas"], dtype=np.float64
    )
    beta = np.asarray(archive["beta"] if "beta" in archive else archive["betas"], dtype=np.float64)
    energy = np.asarray(
        archive["energy"] if "energy" in archive else archive["L"], dtype=np.float64
    )
    qaoa = _qaoa(run_p1)
    gamma_star = float(np.asarray(qaoa["gammas"], dtype=np.float64).ravel()[0])
    beta_star = float(np.asarray(qaoa["betas"], dtype=np.float64).ravel()[0])
    title = f"COBYLA stops at gamma={gamma_star:.2f}, beta={beta_star:.2f}."
    fig, ax = _new_ax(title)
    image = ax.imshow(
        energy,
        origin="lower",
        aspect="auto",
        extent=(float(beta.min()), float(beta.max()), float(gamma.min()), float(gamma.max())),
    )
    fig.colorbar(image, ax=ax, label="energy")
    ax.plot(
        beta_star,
        gamma_star,
        marker="o",
        markersize=10,
        markerfacecolor="white",
        markeredgecolor=COLORS["qaoa_x"],
        markeredgewidth=2,
    )
    ax.set_xlabel("beta")
    ax.set_ylabel("gamma")
    save_figure(fig, "landscape_p1")
    return fig


def _qaoa_frame_ratios(frames_dir: str | Path) -> dict[int, float]:
    root = Path(frames_dir)
    if not root.is_dir():
        return {}
    found: dict[int, float] = {}
    for path in sorted(root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        frame = payload.get("frame")
        ratio = payload.get("ar_top1")
        if ratio is None and isinstance(payload.get("metrics"), dict):
            ratio = payload["metrics"].get("ar_top1")
        if frame is not None and ratio is not None:
            found[int(frame)] = float(ratio)
    return found


def fig_camera_path(path_json: str | Path, frames_dir: str | Path = "cache/path") -> plt.Figure:
    """Greedy and QAOA approximation ratio along the camera path."""
    payload = json.loads(Path(path_json).read_text(encoding="utf-8"))
    frames = sorted(payload["frames"], key=lambda item: int(item["frame"]))
    xs = [int(item["frame"]) for item in frames]
    greedy = [float(item["ar_greedy"]) for item in frames]
    qaoa = _qaoa_frame_ratios(frames_dir)
    below = sum(value < 0.95 for value in greedy)
    if qaoa:
        title = f"Greedy stays below 0.95 on {below} of {len(frames)} frames."
    else:
        title = "QAOA: no data"
    fig, ax = _new_ax(title)
    ax.axhline(1.0, color=COLORS["dp"], linewidth=1.5)
    label_line_end(ax, xs[-1], 1.0, "DP", COLORS["dp"])
    ax.plot(xs, greedy, color=COLORS["greedy"], linewidth=2)
    label_line_end(ax, xs[-1], greedy[-1], "Greedy", COLORS["greedy"])
    ax.fill_between(
        xs,
        greedy,
        0.95,
        where=[value < 0.95 for value in greedy],
        color=COLORS["greedy"],
        alpha=0.18,
        interpolate=True,
    )
    if qaoa:
        paired = [qaoa[frame] for frame in xs if frame in qaoa]
        paired_x = [frame for frame in xs if frame in qaoa]
        ax.plot(paired_x, paired, color=COLORS["qaoa_x"], linewidth=2)
        label_line_end(ax, paired_x[-1], paired[-1], "QAOA", COLORS["qaoa_x"])
    ax.set_xlabel("frame")
    ax.set_ylabel("approximation ratio")
    ax.set_xlim(min(xs), max(xs))
    save_figure(fig, "camera_path")
    return fig


def _note(lines: list[str], name: str, fig: plt.Figure) -> None:
    lines.append(f"`{name}.png` — {fig.axes[0].get_title()}")
    plt.close(fig)


def _runs_for(
    runs: list[dict[str, Any]], scene: str, mixer: str, depth: int | None = None
) -> list[dict[str, Any]]:
    chosen = []
    for run in runs:
        qaoa = run.get("qaoa") or {}
        if run.get("scene") != scene or qaoa.get("mixer") != mixer:
            continue
        if depth is not None and int(qaoa.get("p", -1)) != depth:
            continue
        chosen.append(run)
    return chosen


def main() -> None:
    """Rebuild every figure that has saved data and write figures/README.md."""
    apply_style()
    runs_root = Path("runs")
    runs = load_runs(runs_root) if runs_root.is_dir() else []
    summary_path = runs_root / "summary.csv"
    raw = runs_to_dataframe(runs) if runs else pd.DataFrame()
    if "metrics.ar_top1" in raw.columns:
        table = raw
    elif summary_path.is_file():
        table = pd.read_csv(summary_path)
    else:
        table = raw
    scene = "greedy_trap_1"
    lines: list[str] = []
    has_scene = not table.empty and "scene" in table.columns and (table["scene"] == scene).any()
    if has_scene:
        _note(lines, "ar_vs_p", fig_ar_vs_p(table, scene))
        _note(lines, "feasibility", fig_feasibility(table, scene))
        _note(lines, "amplification", fig_amplification(table, scene))
    else:
        warnings.warn(f"no rows for {scene}; skipping aggregate figures", stacklevel=1)

    top = _runs_for(runs, scene, "x", 3)
    if top:
        _note(lines, "top10", fig_top10(top[0]))
    else:
        warnings.warn(f"no {scene} p=3 X run; skipping top10", stacklevel=1)

    by_seed: dict[Any, list[dict[str, Any]]] = {}
    for run in _runs_for(runs, scene, "x"):
        by_seed.setdefault((run.get("qaoa") or {}).get("seed"), []).append(run)
    history = next((group for group in by_seed.values() if group), None)
    if history:
        _note(lines, "convergence", fig_convergence(history))
    else:
        warnings.warn(f"no {scene} X energy histories; skipping convergence", stacklevel=1)

    landscape = next(
        (
            path
            for path in (Path("cache/landscape_p1.npz"), Path("runs/landscape_p1.npz"))
            if path.is_file()
        ),
        None,
    )
    run_p1 = next(iter(_runs_for(runs, scene, "x", 1)), None)
    if landscape is not None and run_p1 is not None:
        _note(lines, "landscape_p1", fig_landscape_p1(landscape, run_p1))
    else:
        warnings.warn("no p=1 landscape; skipping landscape_p1", stacklevel=1)

    path_json = Path("scenes/greedy_trap_path.json")
    if path_json.is_file():
        _note(lines, "camera_path", fig_camera_path(path_json))
    else:
        warnings.warn(f"missing {path_json}; skipping camera_path", stacklevel=1)

    readme = Path("figures") / "README.md"
    readme.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


if __name__ == "__main__":
    main()
