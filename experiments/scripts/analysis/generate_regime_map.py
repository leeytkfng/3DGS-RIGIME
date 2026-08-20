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
OPT_METHODS = ["Vanilla3DGS", "FSGS"]
METHOD_ABBR = {"Vanilla3DGS": "V3DGS", "FSGS": "FSGS", "MVSplat": "MVSplat", "DepthSplat": "DepthSplat"}
METHOD_COLOR = {"MVSplat": "#4C72B0", "DepthSplat": "#4C72B0", "Vanilla3DGS": "#DD8452", "FSGS": "#55A868"}
METHOD_MARKER = {"MVSplat": "o", "DepthSplat": "o", "Vanilla3DGS": "s", "FSGS": "^"}


def pairs_for(ff_method: str) -> list[tuple[str, str]]:
    return [("Vanilla3DGS", ff_method), ("FSGS", ff_method), ("FSGS", "Vanilla3DGS")]


def tau_for(view_count: int) -> float:
    return 0.5 if view_count in (2, 4) else 1.4


def cell_mean(rows: list[dict], method: str, view_count: int, budget: float) -> dict:
    subset = [r for r in rows if r["method"] == method and r["view_count"] == view_count
              and r["budget"] == budget and r["status"] == "ok"]
    if not subset:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "scene_count": 0}
    return scene_cluster_bootstrap_ci(subset, value_fn=lambda r: r["test_psnr"])


def classify(delta: float, tau: float, method_a: str, method_b: str) -> str:
    """method_a - method_b 부호로 판정. 반환: 실제 승자 method 이름, 'tie', 또는 'no_data'."""
    if np.isnan(delta):
        return "no_data"
    if delta > tau:
        return method_a
    if delta < -tau:
        return method_b
    return "tie"


def build_regime_grid(rows: list[dict], method_a: str, method_b: str) -> np.ndarray:
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
            grid[i, j] = classify(delta, tau, method_a, method_b)
    return grid


TIE_COLOR = "#DDDDDD"
NO_DATA_COLOR = "#FFFFFF"


def plot_regime_map(rows: list[dict], out_dir: Path, progress_note: str, ff_method: str, dataset_label: str) -> None:
    pairs = pairs_for(ff_method)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.4), sharey=True)
    for ax, (method_a, method_b) in zip(axes, pairs):
        grid = build_regime_grid(rows, method_a, method_b)
        color_grid = np.zeros((len(VIEW_COUNTS), len(BUDGETS), 3))
        for i in range(len(VIEW_COUNTS)):
            for j in range(len(BUDGETS)):
                winner = grid[i, j]
                if winner == "tie":
                    hexcolor = TIE_COLOR
                elif winner == "no_data":
                    hexcolor = NO_DATA_COLOR
                else:
                    hexcolor = METHOD_COLOR[winner]
                color_grid[i, j] = matplotlib.colors.to_rgb(hexcolor)
        ax.imshow(color_grid, aspect="auto")
        for i in range(len(VIEW_COUNTS)):
            for j in range(len(BUDGETS)):
                winner = grid[i, j]
                if winner == "tie":
                    label, textcolor = "=", "black"
                elif winner == "no_data":
                    label, textcolor = "", "black"
                else:
                    label, textcolor = METHOD_ABBR.get(winner, winner), "white"
                if label:
                    ax.text(j, i, label, ha="center", va="center", fontsize=9, color=textcolor,
                             fontweight="bold")
        ax.set_xticks(range(len(BUDGETS)))
        ax.set_xticklabels([f"{int(b)}s" for b in BUDGETS])
        ax.set_yticks(range(len(VIEW_COUNTS)))
        ax.set_yticklabels([f"{v}-view" for v in VIEW_COUNTS])
        ax.set_title(f"{METHOD_ABBR.get(method_a, method_a)} vs {METHOD_ABBR.get(method_b, method_b)}", fontsize=10)
        ax.set_xlabel("budget")
    axes[0].set_ylabel("view count")

    legend_methods = [ff_method] + OPT_METHODS
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=METHOD_COLOR[m], label=METHOD_ABBR.get(m, m))
        for m in legend_methods
    ]
    handles.append(plt.Rectangle((0, 0), 1, 1, color=TIE_COLOR, label="Tie"))
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=NO_DATA_COLOR, edgecolor="black", label="No data"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, -0.05), fontsize=9)
    fig.suptitle(f"C1-a Regime Map ({dataset_label}, τ=0.5dB@2/4-view, 1.4dB@8/12-view) — {progress_note}", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(out_dir / "regime_map.png", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / "regime_map.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_dir / 'regime_map.png'}")


def plot_pareto_frontier(rows: list[dict], out_dir: Path, progress_note: str, ff_method: str, dataset_label: str) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.3), sharex=False)

    for ax, vc in zip(axes.flat, VIEW_COUNTS):
        for method in [ff_method] + OPT_METHODS:
            xs, ys, ylo, yhi = [], [], [], []
            for b in BUDGETS:
                stat = cell_mean(rows, method, vc, b)
                if stat["scene_count"] == 0:
                    continue
                xs.append(b)
                ys.append(stat["mean"])
                # n=1 scene bootstrap can yield ci bounds that cross the mean by float noise
                ylo.append(max(0.0, stat["mean"] - stat["ci_low"]))
                yhi.append(max(0.0, stat["ci_high"] - stat["mean"]))
            if not xs:
                continue
            ax.errorbar(xs, ys, yerr=[ylo, yhi], marker=METHOD_MARKER[method], color=METHOD_COLOR[method],
                        label=method, capsize=3, linewidth=1.5, markersize=5)
        ax.set_xscale("log")
        ax.set_title(f"{vc}-view", fontsize=10)
        ax.set_xlabel("wall-clock budget (s)")
        ax.set_ylabel("test PSNR (dB)")

    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"C1-a Quality-Time Pareto Frontier ({dataset_label}) — {progress_note}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_dir / "pareto_frontier.png", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / "pareto_frontier.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_dir / 'pareto_frontier.png'}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default=str(REPO_ROOT / "experiments/outputs/re10k_c1a_main/c1a_main_summary.json"))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "experiments/docs/paper/overleaf_draft/figures"))
    parser.add_argument("--total-combos", type=int, default=len(VIEW_COUNTS) * 30, help="30 scenes x 4 view_counts = 120")
    parser.add_argument("--ff-method", default="MVSplat", help="feed-forward method name as it appears in the summary rows (MVSplat for RE10K, DepthSplat for DL3DV)")
    parser.add_argument("--dataset-label", default="RE10K", help="label used in figure titles")
    args = parser.parse_args()

    rows = json.loads(Path(args.summary).read_text())
    combos_done = len({(r["scene"], r["view_count"]) for r in rows})
    progress_note = (
        f"PREVIEW, {combos_done}/{args.total_combos} combos ({100*combos_done/args.total_combos:.0f}%)"
        if combos_done < args.total_combos else "final, all combos complete"
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_regime_map(rows, out_dir, progress_note, args.ff_method, args.dataset_label)
    plot_pareto_frontier(rows, out_dir, progress_note, args.ff_method, args.dataset_label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
