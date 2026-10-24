"""Check that the figures and the README results are up to date with runs/summary.csv.

Every file named in figures/README.md must exist and be newer than runs/summary.csv, and
README.md must contain the RESULTS block. Exits with code 1 on any problem.
"""

import argparse
import re
import sys
from pathlib import Path

START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"
FILE_RE = re.compile(r"`([^`\s]+\.[A-Za-z0-9]+)`")


def figure_files(notes: Path) -> list[Path]:
    """Files named in backticks in figures/README.md, relative to its folder."""
    names = FILE_RE.findall(notes.read_text(encoding="utf-8"))
    return [notes.parent / name for name in dict.fromkeys(names)]


def check_figures(notes: Path, summary: Path) -> list[str]:
    if not notes.is_file():
        return [f"{notes} is missing: run `make figures`"]
    if not summary.is_file():
        return [f"{summary} is missing: run `make experiments`"]
    files = figure_files(notes)
    if not files:
        return [f"{notes} names no figure files"]
    summary_mtime = summary.stat().st_mtime
    errors = []
    for path in files:
        if not path.is_file():
            errors.append(f"{path} is listed in {notes} but does not exist")
        elif path.stat().st_mtime <= summary_mtime:
            errors.append(f"{path} is older than {summary}: run `make figures`")
    return errors


def check_readme(readme: Path) -> list[str]:
    if not readme.is_file():
        return [f"{readme} is missing"]
    text = readme.read_text(encoding="utf-8")
    start, end = text.find(START), text.find(END)
    if start < 0 or end < 0:
        return [f"{readme} has no {START} ... {END} block"]
    if end < start:
        return [f"{readme}: {END} comes before {START}"]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Check figures and README before a release")
    parser.add_argument("--notes", type=Path, default=Path("figures/README.md"))
    parser.add_argument("--summary", type=Path, default=Path("runs/summary.csv"))
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    args = parser.parse_args()

    errors = check_figures(args.notes, args.summary) + check_readme(args.readme)
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    if errors:
        sys.exit(1)
    print(f"ok: {len(figure_files(args.notes))} figures are newer than {args.summary}")


if __name__ == "__main__":
    main()
