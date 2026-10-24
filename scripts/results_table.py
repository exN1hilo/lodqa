"""Markdown results table for one scene: DP, Greedy, Random and QAOA by mixer and p."""

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from lodq.storage import load_runs

START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"
COLUMNS = (
    ("ar_top1", "AR top-1"),
    ("p_feas", "P_feas"),
    ("budget_util_best", "Budget use"),
    ("amplification", "P_opt gain"),
)
MIXERS = (("x", "QAOA-X"), ("xy", "QAOA-XY"))

Row = tuple[str, list[tuple[float, float]]]


def _fmt(mean: float, std: float) -> str:
    if math.isnan(mean):
        return "—"
    if math.isnan(std):
        return f"{mean:.3f}"
    return f"{mean:.3f} ± {std:.3f}"


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    arr = np.asarray(values, dtype=np.float64)
    # ddof=1 like pandas in summary.csv; a single run has no spread
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else math.nan


def classical_rows(runs: list[dict], scene: str) -> list[Row]:
    """DP, Greedy and Random from the run records of one scene.

    DP and Greedy are deterministic: P_feas = 1, and P_opt is 1 for DP and 1 for Greedy
    only when it finds the DP choice. Their gain is P_opt divided by the uniform P_opt.
    """
    values: dict[str, dict[str, list[float]]] = {
        name: {key: [] for key, _ in COLUMNS} for name in ("DP", "Greedy", "Random")
    }
    for run in runs:
        if run.get("scene") != scene or "dp" not in run:
            continue
        dp, greedy, rand = run["dp"], run.get("greedy"), run.get("random")
        dW = float(run["problem"]["dW"])
        p_opt_random = float(rand["p_opt"]) if rand else math.nan

        def add(name: str, ar: float, p_feas: float, util: float, amp: float) -> None:
            for (key, _), value in zip(COLUMNS, (ar, p_feas, util, amp), strict=True):
                values[name][key].append(value)

        add("DP", 1.0, 1.0, dp["used"] / dW, 1.0 / p_opt_random)
        if greedy is not None:
            hit = float(list(greedy["choice"]) == list(dp["choice"]))
            add(
                "Greedy",
                greedy["value"] / dp["value"],
                1.0,
                greedy["used"] / dW,
                hit / p_opt_random,
            )
        if rand is not None:
            add("Random", *(float(rand[key]) for key, _ in COLUMNS))
    return [
        (name, [_mean_std(cols[key]) for key, _ in COLUMNS])
        for name, cols in values.items()
        if cols["ar_top1"]
    ]


def qaoa_rows(summary: pd.DataFrame, scene: str) -> list[Row]:
    frame = summary[summary["scene"] == scene]
    rows: list[Row] = []
    for mixer, label in MIXERS:
        part = frame[frame["qaoa.mixer"] == mixer].sort_values("qaoa.p")
        for _, record in part.iterrows():
            cells = [
                (
                    float(record.get(f"metrics.{key}_mean", math.nan)),
                    float(record.get(f"metrics.{key}_std", math.nan)),
                )
                for key, _ in COLUMNS
            ]
            rows.append((f"{label} p={int(record['qaoa.p'])}", cells))
    return rows


def markdown_table(rows: list[Row]) -> str:
    header = ["Method", *(title for _, title in COLUMNS)]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---", *("---:" for _ in COLUMNS)]) + "|",
    ]
    for name, cells in rows:
        lines.append("| " + " | ".join([name, *(_fmt(m, s) for m, s in cells)]) + " |")
    return "\n".join(lines)


def write_readme(path: Path, table: str, scene: str) -> None:
    block = f"{START}\n\nScene `{scene}`, mean ± std over runs.\n\n{table}\n\n{END}"
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.DOTALL)
    if pattern.search(text):
        text = pattern.sub(lambda _: block, text, count=1)
    else:
        text = text.rstrip("\n") + f"\n\n## Results\n\n{block}\n"
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the results table for one scene")
    parser.add_argument("--scene", default="greedy_trap_1")
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--write-readme", action="store_true")
    args = parser.parse_args()

    summary_path = args.runs_root / "summary.csv"
    if not summary_path.is_file() or summary_path.stat().st_size == 0:
        raise SystemExit(f"{summary_path} is missing or empty: run `make experiments` first")
    summary = pd.read_csv(summary_path)
    rows = classical_rows(load_runs(args.runs_root), args.scene) + qaoa_rows(summary, args.scene)
    if not rows:
        raise SystemExit(f"no results for scene {args.scene!r} in {args.runs_root}")

    table = markdown_table(rows)
    print(table)
    if args.write_readme:
        write_readme(args.readme, table, args.scene)
        print(f"\nupdated {args.readme}")


if __name__ == "__main__":
    main()
