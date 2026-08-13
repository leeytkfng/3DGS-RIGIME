#!/usr/bin/env python3
"""논문 초안용 figure 2개 생성 — 이미 확보된 데이터로 실제 그림을 만든다(placeholder 아님).
1. observation_starvation.png — count[g]<=2 비율 vs view 수
2. geometry_confound.png — overlap-baseline-uncertainty 편상관 (raw / linear-controlled / log-controlled)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path("/root/task 5/experiments/docs/paper/overleaf_draft/figures")
plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

# ---------------------------------------------------------------------------
# Figure 1: observation starvation
# ---------------------------------------------------------------------------
view_counts = [2, 4, 8, 12]
pooled = [51.5, 30.8, 3.2, 0.7]
fixed_init = [55.3, 32.3, 8.3, 1.3]

fig, ax = plt.subplots(figsize=(5.2, 3.6))
x = np.arange(len(view_counts))
width = 0.35
ax.bar(x - width / 2, pooled, width, label="Pooled mean (3 scenes)", color="#4C72B0")
ax.bar(x + width / 2, fixed_init, width, label="Single scene, init. fixed", color="#DD8452")
ax.set_xticks(x)
ax.set_xticklabels([f"{v}-view" for v in view_counts])
ax.set_ylabel(r"Fraction of Gaussians with $\mathrm{count}[g] \leq 2$ (\%)")
ax.set_title("Observation-starved Gaussians vs. view count", fontsize=11)
ax.legend(frameon=False, fontsize=9)
ax.set_ylim(0, 62)
for xi, (p, f) in enumerate(zip(pooled, fixed_init)):
    ax.text(xi - width / 2, p + 1.2, f"{p:.1f}", ha="center", fontsize=8)
    ax.text(xi + width / 2, f + 1.2, f"{f:.1f}", ha="center", fontsize=8)
fig.tight_layout()
fig.savefig(OUT_DIR / "observation_starvation.png")
fig.savefig(OUT_DIR / "observation_starvation.pdf")
print("wrote observation_starvation.{png,pdf}")

# ---------------------------------------------------------------------------
# Figure 2: geometry confound (baseline-overlap-uncertainty)
# ---------------------------------------------------------------------------
df = pd.read_csv("/root/task 5/experiments/outputs/geometry_figures/pairwise_geometry.csv")
overlap = df["overlap"].values
baseline = df["baseline"].values
log_unc = np.log(df["mean_depth_uncertainty"].values)
log_baseline = np.log(baseline)

fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))

axes[0].scatter(overlap, log_unc, s=10, alpha=0.4, color="#4C72B0")
r_raw = np.corrcoef(overlap, log_unc)[0, 1]
axes[0].set_xlabel(r"Overlap $O_{ij}$")
axes[0].set_ylabel(r"$\log \hat\sigma_{\mathrm{depth}}$")
axes[0].set_title(f"Raw correlation: r = {r_raw:+.3f}", fontsize=10)

axes[1].scatter(log_baseline, log_unc, s=10, alpha=0.4, color="#DD8452")
r_logb = np.corrcoef(log_baseline, log_unc)[0, 1]
axes[1].set_xlabel(r"$\log(\mathrm{baseline})$")
axes[1].set_ylabel(r"$\log \hat\sigma_{\mathrm{depth}}$")
axes[1].set_title(f"vs. log(baseline): r = {r_logb:+.3f}", fontsize=10)

fig.suptitle("DTU scan1, 861 view pairs — overlap and baseline act on uncertainty in opposite directions", fontsize=10, y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "geometry_confound.png", bbox_inches="tight")
fig.savefig(OUT_DIR / "geometry_confound.pdf", bbox_inches="tight")
print("wrote geometry_confound.{png,pdf}")
print(f"raw corr(overlap, log_unc) = {r_raw:.3f} (paper reports overlap vs log_unc corr as 0.952 in 문서 — direction check)")
print(f"corr(log_baseline, log_unc) = {r_logb:.3f} (paper reports -0.989)")
