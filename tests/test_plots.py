import warnings
from pathlib import Path

import pandas as pd

from lodq.plots import main


def test_main_writes_readme_and_skips_missing_runs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs").mkdir()
    rows = []
    for depth in (1, 2, 3, 4):
        for mixer, ratio in (("x", 0.2 * depth), ("xy", 0.15 * depth)):
            rows.append(
                {
                    "scene": "greedy_trap_1",
                    "qaoa.mixer": mixer,
                    "qaoa.p": depth,
                    "metrics.ar_top1_mean": ratio,
                    "metrics.ar_top1_std": 0.01,
                    "metrics.p_feas_mean": ratio,
                    "metrics.p_feas_std": 0.0,
                    "metrics.p_onehot_mean": ratio,
                    "metrics.p_onehot_std": 0.0,
                    "metrics.amplification_mean": 10**depth,
                    "metrics.amplification_std": 0.0,
                }
            )
    pd.DataFrame(rows).to_csv(tmp_path / "runs" / "summary.csv", index=False)
    (tmp_path / "scenes").mkdir()
    frames = ", ".join(
        f'{{"frame": {index}, "ar_greedy": {0.9 if index > 2 else 1.0}}}' for index in range(4)
    )
    (tmp_path / "scenes" / "greedy_trap_path.json").write_text(
        f'{{"scene": "greedy_trap_1", "frames": [{frames}]}}',
        encoding="utf-8",
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        main()

    readme = (tmp_path / "figures" / "README.md").read_text(encoding="utf-8")
    assert "`ar_vs_p.png` — " in readme
    assert "`feasibility.png` — " in readme
    assert "`amplification.png` — " in readme
    assert "`camera_path.png` — " in readme
    assert (tmp_path / "figures" / "ar_vs_p.svg").is_file()
    assert any("top10" in str(item.message) for item in caught)
    assert any("landscape" in str(item.message) for item in caught)
