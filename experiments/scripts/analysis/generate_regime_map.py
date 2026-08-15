#!/usr/bin/env python3
"""C1-a Regime Map + Pareto frontier figure 생성 (§4 Figure 우선순위 ①②).

`run_re10k_c1a_main.py`가 쓰는 `c1a_main_summary.json`을 읽어서:
1. regime_map.png — 방법 쌍(pair)별 view_count x budget 승패 지도(3-class: FF/OPT/Tie),
   §5.5 판정식(Δ=PSNR_FF-PSNR_OPT, τ=view_count-tiered 0.5/1.4dB)과 §5.12 Holm family
   정의(방법 쌍마다 별도 family)를 그대로 따른다. scene 평균은 §protocol_utils의
   `scene_cluster_bootstrap_ci`로 낸다(seed는 반복측정이라 scene만 독립 단위).
2. pareto_frontier.png — view_count별 품질(PSNR)-시간(wall-clock) 곡선, 세 방법 비교.

본 실험이 아직 진행 중이면(§120 combo 중 일부만 완료) 그 시점까지 확보된 데이터로
PREVIEW를 찍고 제목에 진행률을 표시한다 — 완료 후 같은 커맨드로 재실행하면 최종판이 된다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from protocol_utils import scene_cluster_bootstrap_ci  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
VIEW_COUNTS = [2, 4, 8, 12]
BUDGETS = [1.0, 10.0, 60.0, 300.0]
FF_METHOD = "MVSplat"
OPT_METHODS = ["Vanilla3DGS", "FSGS"]
PAIRS = [("Vanilla3DGS", "MVSplat"), ("FSGS", "MVSplat"), ("FSGS", "Vanilla3DGS")]


def tau_for(view_count: int) -> float:
    return 0.5 if view_count in (2, 4) else 1.4


def cell_mean(rows: list[dict], method: str, view_count: int, budget: float) -> dict:
    subset = [r for r in rows if r["method"] == method and r["view_count"] == view_count
              and r["budget"] == budget and r["status"] == "ok"]
    if not subset:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "scene_count": 0}
    return scene_cluster_bootstrap_ci(subset, value_fn=lambda r: r["test_psnr"])


def classify(delta: float, tau: float) -> str:
    if np.isnan(delta):
        return "no_data"
    if delta > tau:
        return "a_wins"
    if delta < -tau:
        return "b_wins"
    return "tie"


def build_regime_grid(rows: list[dict], method_a: str, method_b: str) -> np.ndarray:
    """method_a - method_b 부호로 판정. 반환: view_count x budget 문자열 배열."""
    grid = np.empty((len(VIEW_COUNTS), len(BUDGETS)), dtype=object)
    for i, vc in enumerate(VIEW_COUNTS):
        tau = tau_for(vc)
        for j, b in enumerate(BUDGETS):
            stat_a = cell_mean(rows, method_a, vc, b)
            stat_b = cell_mean(rows, method_b, vc, b)
            if stat_a["scene_count"] == 0 or stat_b["scene_count"] == 0:
                grid[i, j] = "no_data"
                continue
            delta = stat_a["mean"] - stat_b["mean"]
            grid[i, j] = classify(delta, tau)
    return grid


CLASS_COLOR = {
    "a_wins": "#4C72B0",
    "b_wins": "#DD8452",
    "tie": "#DDDDDD",
    "no_data": "#FFFFFF",
}
CLASS_TEXT = {"a_wins": "A", "b_wins": "B", "tie": "="}


def plot_regime_map(rows: list[dict], out_dir: Path, progress_note: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.4), sharey=True)
    for ax, (method_a, method_b) in zip(axes, PAIRS):
        grid = build_regime_grid(rows, method_a, method_b)
        color_grid = np.zeros((len(VIEW_COUNTS), len(BUDGETS), 3))
        for i in range(len(VIEW_COUNTS)):
            for j in range(len(BUDGETS)):
                hexcolor = CLASS_COLOR[grid[i, j]]
                color_grid[i, j] = matplotlib.colors.to_rgb(hexcolor)
        ax.imshow(color_grid, aspect="auto")
        for i in range(len(VIEW_COUNTS)):
            for j in range(len(BUDGETS)):
                label = CLASS_TEXT.get(grid[i, j], "")
                if label:
                    ax.text(j, i, label, ha="center", va="center", fontsize=10, color="black")
        ax.set_xticks(range(len(BUDGETS)))
        ax.set_xticklabels([f"{int(b)}s" for b in BUDGETS])
        ax.set_yticks(range(len(VIEW_COUNTS)))
        ax.set_yticklabels([f"{v}-view" for v in VIEW_COUNTS])
        ax.set_title(f"{method_a} (A) vs {method_b} (B)", fontsize=10)
        ax.set_xlabel("budget")
    axes[0].set_ylabel("view count")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=CLASS_COLOR["a_wins"], label="A wins"),
        plt.Rectangle((0, 0), 1, 1, color=CLASS_COLOR["b_wins"], label="B wins"),
        plt.Rectangle((0, 0), 1, 1, color=CLASS_COLOR["tie"], label="Tie"),
        plt.Rectangle((0, 0), 1, 1, facecolor=CLASS_COLOR["no_data"], edgecolor="black", label="No data"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.05), fontsize=9)
    fig.suptitle(f"C1-a Regime Map (RE10K, τ=0.5dB@2/4-view, 1.4dB@8/12-view) — {progress_note}", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(out_dir / "regime_map.png", dpi=150)
    fig.savefig(out_dir / "regime_map.pdf")
    plt.close(fig)
    print(f"wrote {out_dir / 'regime_map.png'}")


def plot_pareto_frontier(rows: list[dict], out_dir: Path, progress_note: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 7.0), sharex=False)
    colors = {"MVSplat": "#4C72B0", "Vanilla3DGS": "#DD8452", "FSGS": "#55A868"}
    markers = {"MVSplat": "o", "Vanilla3DGS": "s", "FSGS": "^"}

    for ax, vc in zip(axes.flat, VIEW_COUNTS):
        for method in [FF_METHOD] + OPT_METHODS:
            xs, ys, ylo, yhi = [], [], [], []
            for b in BUDGETS:
                stat = cell_mean(rows, method, vc, b)
                if stat["scene_count"] == 0:
                    continue
                xs.append(b)
                ys.append(stat["mean"])
                ylo.append(stat["mean"] - stat["ci_low"])
                yhi.append(stat["ci_high"] - stat["mean"])
            if not xs:
                continue
            ax.errorbar(xs, ys, yerr=[ylo, yhi], marker=markers[method], color=colors[method],
                        label=method, capsize=3, linewidth=1.5, markersize=5)
        ax.set_xscale("log")
        ax.set_title(f"{vc}-view", fontsize=10)
        ax.set_xlabel("wall-clock budget (s)")
        ax.set_ylabel("test PSNR (dB)")

    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"C1-a Quality-Time Pareto Frontier (RE10K) — {progress_note}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "pareto_frontier.png", dpi=150)
    fig.savefig(out_dir / "pareto_frontier.pdf")
    plt.close(fig)
    print(f"wrote {out_dir / 'pareto_frontier.png'}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default=str(REPO_ROOT / "experiments/outputs/re10k_c1a_main/c1a_main_summary.json"))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "experiments/docs/paper/overleaf_draft/figures"))
    parser.add_argument("--total-combos", type=int, default=len(VIEW_COUNTS) * 30, help="30 scenes x 4 view_counts = 120")
    args = parser.parse_args()

    rows = json.loads(Path(args.summary).read_text())
    combos_done = len({(r["scene"], r["view_count"]) for r in rows})
    progress_note = (
        f"PREVIEW, {combos_done}/{args.total_combos} combos ({100*combos_done/args.total_combos:.0f}%)"
        if combos_done < args.total_combos else "final, all combos complete"
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_regime_map(rows, out_dir, progress_note)
    plot_pareto_frontier(rows, out_dir, progress_note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
