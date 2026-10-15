#!/usr/bin/env python3
"""Plot current-phase training-only expanding-window trial metrics."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "smart-canteen-mpl")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "trial_total_mape_net_benefit.png"
SAMPLE_OUTPUT = ROOT / "sample_efficiency.png"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _font_name() -> str:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "Noto Sans CJK SC"):
        if name in available:
            return name
    return "DejaVu Sans"


def load_trials() -> pd.DataFrame:
    records = _read_jsonl(ROOT / "trials.jsonl")
    corrections = {
        int(row["trial_index"]): row
        for row in _read_jsonl(ROOT / "selection_corrections.jsonl")
    }
    for record in records:
        correction = corrections.get(int(record["trial_index"]))
        if correction:
            record["is_new_best"] = correction["is_new_best"]
    frame = pd.DataFrame(records)
    required = {"trial_index", "total_mape", "net_benefit", "is_new_best"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"实验记录缺少字段: {sorted(missing)}")
    return frame.sort_values("trial_index").drop_duplicates("trial_index", keep="last")


def main() -> None:
    trials = load_trials()
    font = _font_name()
    plt.rcParams.update({"font.family": font, "axes.unicode_minus": False})

    fig, axes = plt.subplots(2, 1, figsize=(14, 8.5), sharex=True, constrained_layout=True)
    accepted = trials[trials["is_new_best"].fillna(False).astype(bool)]

    axes[0].plot(
        trials["trial_index"], trials["total_mape"], color="#1769aa", marker="o",
        markersize=3.5, linewidth=1.4, label="训练内滚动回测",
    )
    axes[0].scatter(
        accepted["trial_index"], accepted["total_mape"], color="#2e8b57", marker="o",
        s=30, zorder=3, label="当轮被接受",
    )
    axes[0].set_ylabel("Total MAPE (%)")
    axes[0].set_title("每轮 Total MAPE（越低越好）")

    axes[1].plot(
        trials["trial_index"], trials["net_benefit"] / 10_000, color="#d1495b", marker="s",
        markersize=3.2, linewidth=1.4, label="训练内滚动回测",
    )
    axes[1].scatter(
        accepted["trial_index"], accepted["net_benefit"] / 10_000, color="#2e8b57", marker="o",
        s=30, zorder=3, label="当轮被接受",
    )
    axes[1].set_ylabel("Net benefit（万元）")
    axes[1].set_xlabel("Trial")
    axes[1].set_title("每轮 Net benefit（越高越好）")

    for axis in axes:
        axis.grid(axis="y", color="#d8d8d8", linewidth=0.7, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(loc="best", frameon=False)

    fig.suptitle(
        "因子与菜单容量阶段：每轮训练集 expanding-window 回测",
        fontsize=16,
    )
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    fig.savefig(SAMPLE_OUTPUT, dpi=180, bbox_inches="tight")
    print(OUTPUT)


if __name__ == "__main__":
    main()
