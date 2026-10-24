"""Запись и чтение прогонов в runs/ (T02)."""

import json
import re
import subprocess
from collections.abc import Collection
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ARRAY_SUFFIX = "_array"
TAG_RE = re.compile(r"[A-Za-z0-9._-]+")
RUN_DIR_RE = re.compile(r"\d{8}-\d{6}-(?P<tag>.+)")
META_KEYS = ("tag", "git_hash", "timestamp")


class NumpyEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def git_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def _split_arrays(data: dict, prefix: str = "") -> tuple[dict, dict[str, np.ndarray]]:
    clean: dict = {}
    arrays: dict[str, np.ndarray] = {}
    for key, value in data.items():
        path = f"{prefix}{key}"
        if isinstance(key, str) and key.endswith(ARRAY_SUFFIX):
            arr = np.asarray(value)
            if arr.dtype.kind not in "biufc":
                raise TypeError(f"{path}: массив должен быть числовым")
            arrays[path] = arr
        elif isinstance(value, dict):
            clean[key], nested = _split_arrays(value, f"{path}.")
            arrays.update(nested)
        else:
            clean[key] = value
    return clean, arrays


def _merge_arrays(run: dict, arrays: dict[str, np.ndarray]) -> dict:
    for path, arr in arrays.items():
        *parents, key = path.split(".")
        node = run
        for name in parents:
            node = node.setdefault(name, {})
        node[key] = arr
    return run


def _run_dirs(root: str | Path) -> list[tuple[str, Path]]:
    root = Path(root)
    if not root.is_dir():
        return []
    found = []
    for run_dir in sorted(root.iterdir()):
        match = RUN_DIR_RE.fullmatch(run_dir.name)
        if match and (run_dir / "run.json").is_file():
            found.append((match["tag"], run_dir))
    return found


def save_run(run: dict, tag: str, root: str | Path = "runs") -> Path:
    """Пишет <root>/<YYYYmmdd-HHMMSS>-<tag>/run.json (+ arrays.npz) и возвращает папку прогона."""
    if not TAG_RE.fullmatch(tag):
        raise ValueError(f"недопустимый tag {tag!r}: только латиница, цифры и символы . _ -")
    now = datetime.now().astimezone()
    meta = {"tag": tag, "git_hash": git_hash(), "timestamp": now.isoformat(timespec="seconds")}
    data, arrays = _split_arrays(run)
    payload = meta | {k: v for k, v in data.items() if k not in META_KEYS}
    text = json.dumps(payload, cls=NumpyEncoder, ensure_ascii=False, indent=2)

    run_dir = Path(root) / f"{now:%Y%m%d-%H%M%S}-{tag}"
    run_dir.mkdir(parents=True)
    if arrays:
        np.savez_compressed(run_dir / "arrays.npz", **arrays)
    tmp = run_dir / "run.json.tmp"
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(run_dir / "run.json")
    return run_dir


def load_runs(root: str | Path = "runs", tag_prefix: str | None = None) -> list[dict]:
    runs = []
    for tag, run_dir in _run_dirs(root):
        if tag_prefix is not None and not tag.startswith(tag_prefix):
            continue
        run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        npz_path = run_dir / "arrays.npz"
        if npz_path.is_file():
            with np.load(npz_path) as npz:
                run = _merge_arrays(run, {name: npz[name] for name in npz.files})
        runs.append(run)
    return runs


def run_exists(root: str | Path, tag: str) -> bool:
    return any(found == tag for found, _ in _run_dirs(root))


def _flatten(data: dict, keep: Collection[str], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict) and key not in keep:
            flat |= _flatten(value, keep, f"{name}.")
        else:
            flat[name] = value
    return flat


def runs_to_dataframe(runs: list[dict], keep: Collection[str] = ("counts",)) -> pd.DataFrame:
    """Одна строка на прогон; вложенные dict → колонки через точку, кроме ключей из keep."""
    return pd.DataFrame([_flatten(run, keep) for run in runs])
