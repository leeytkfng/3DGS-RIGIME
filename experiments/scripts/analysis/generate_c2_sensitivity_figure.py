#!/usr/bin/env python3
"""C2 depth-noise sensitivity figure 생성 (§V.4, H1 검증).

`run_c2_sensitivity_sweep.py`가 쓰는 `c2_sensitivity_summary.json`(640 combo, 2560 rows)을
읽어서 2x2 그림을 만든다 — 위 행은 개별 노이즈($\sigma$) 축, 아래 행은 전역 스케일 편향($s$)
축, 왼쪽 열은 test PSNR, 오른쪽 열은 Gaussian 수. 각 패널에 대표 조건 4개를 선으로 겹쳐 그린다.
`generate_regime_map.py`와 같은 패턴(scene cluster bootstrap CI, errorbar)을 따른다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from protocol_utils import scene_cluster_bootstrap_ci  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
NOISE_LEVELS = [0.0, 0.01, 0.03, 0.05, 0.10]
SCALE_LEVELS = [0.9, 0.95, 1.0, 1.05, 1.1]
CONDITIONS = ["2view_low_overlap", "4view_low_overlap", "4view_high_overlap", "12view_high_overlap"]
COND_LABEL = {
    "2view_low_overlap": "2-view, low overlap",
    "4view_low_overlap": "4-view, low overlap",
    "4view_high_overlap": "4-view, high overlap",
    "12view_high_overlap": "12-view, high overlap",
}
# seaborn "deep" 팔레트 — generate_regime_map.py의 METHOD_COLOR와 같은 계열(파랑/주황/초록) + 보라.
COND_COLOR = {
    "2view_low_overlap": "#4C72B0",
    "4view_low_overlap": "#DD8452",
    "4view_high_overlap": "#55A868",
    "12view_high_overlap": "#8172B2",
}
COND_MARKER = {
    "2view_low_overlap": "o",
    "4view_low_overlap": "s",
    "4view_high_overlap": "^",
    "12view_high_overlap": "D",
}


def cell_stat(rows: list[dict], condition: str, phase: str, level_field: str, level: float, value_field: str) -> dict:
    subset = [
        r for r in rows
        if r["condition"] == condition and r["phase"] == phase and r["budget"] == 300.0
        and r["status"] == "ok" and abs(r[level_field] - level) < 1e-9
    ]
    if not subset:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "scene_count": 0}
    return scene_cluster_bootstrap_ci(subset, value_fn=lambda r: r[value_field])


def plot_axis(ax_psnr, ax_gauss, rows: list[dict], phase: str, level_field: str, levels: list[float], xlabel: str) -> None:
    for cond in CONDITIONS:
        xs, psnr_y, psnr_lo, psnr_hi = [], [], [], []
        gauss_y, gauss_lo, gauss_hi = [], [], []
        for level in levels:
            s_psnr = cell_stat(rows, cond, phase, level_field, level, "test_psnr")
            s_gauss = cell_stat(rows, cond, phase, level_field, level, "gaussian_count")
            if s_psnr["scene_count"] == 0:
                continue
            xs.append(level)
            psnr_y.append(s_psnr["mean"])
            psnr_lo.append(max(0.0, s_psnr["mean"] - s_psnr["ci_low"]))
            psnr_hi.append(max(0.0, s_psnr["ci_high"] - s_psnr["mean"]))
            gauss_y.append(s_gauss["mean"])
            gauss_lo.append(max(0.0, s_gauss["mean"] - s_gauss["ci_low"]))
            gauss_hi.append(max(0.0, s_gauss["ci_high"] - s_gauss["mean"]))
        if not xs:
            continue
        style = dict(color=COND_COLOR[cond], marker=COND_MARKER[cond], label=COND_LABEL[cond],
                     capsize=3, linewidth=1.5, markersize=5)
        ax_psnr.errorbar(xs, psnr_y, yerr=[psnr_lo, psnr_hi], **style)
        ax_gauss.errorbar(xs, gauss_y, yerr=[gauss_lo, gauss_hi], **style)

    ax_psnr.set_xlabel(xlabel)
    ax_psnr.set_ylabel("test PSNR (dB)")
    ax_gauss.set_xlabel(xlabel)
    ax_gauss.set_ylabel("Gaussian count")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default=str(REPO_ROOT / "experiments/outputs/c2_sensitivity_sweep/c2_sensitivity_summary.json"))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "experiments/docs/paper/overleaf_draft/figures"))
    args = parser.parse_args()

    rows = json.loads(Path(args.summary).read_text())
    combos_done = len({(r["scene"], r["condition"], r["phase"], r["sigma"], r["scale_bias"], r["seed"]) for r in rows})
    total_combos = 640
    progress_note = (
        f"PREVIEW, {combos_done}/{total_combos} combos ({100*combos_done/total_combos:.0f}%)"
        if combos_done < total_combos else "final, 640/640 combos complete"
    )

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 7.0))
    plot_axis(axes[0, 0], axes[0, 1], rows, "c2_depth_noise", "sigma", NOISE_LEVELS, r"depth noise $\sigma$")
    plot_axis(axes[1, 0], axes[1, 1], rows, "c2_depth_scale_bias", "scale_bias", SCALE_LEVELS, r"global scale bias $s$")
    axes[0, 0].set_title(r"PSNR vs. $\sigma$ (scale bias fixed at 1.0)", fontsize=10)
    axes[0, 1].set_title(r"Gaussian count vs. $\sigma$", fontsize=10)
    axes[1, 0].set_title(r"PSNR vs. $s$ (noise fixed at 0)", fontsize=10)
    axes[1, 1].set_title(r"Gaussian count vs. $s$", fontsize=10)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="best")

    fig.suptitle(f"C2 Depth-Noise Sensitivity (DTU 8-scan, budget=300s) — {progress_note}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "c2_depth_sensitivity.png", dpi=150)
    fig.savefig(out_dir / "c2_depth_sensitivity.pdf")
    plt.close(fig)
    print(f"wrote {out_dir / 'c2_depth_sensitivity.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
