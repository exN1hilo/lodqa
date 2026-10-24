"""Figures rebuilt from saved runs, without launching QAOA again."""

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
