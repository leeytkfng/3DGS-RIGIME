#!/usr/bin/env python3
"""H2 검증 — "view 수가 적을수록 optimization 품질 정점이 빨리 오고, 정점 이후 하강이
가파르다"(main.tex 표 hypotheses)를 RE10K C1-a 궤적 로그로 검정한다.

각 (scene, view_count, seed) trajectory는 budget snapshot 4개(1/10/60/300s)뿐인
성긴(sparse) 로그다 — 연속적인 "정점 시각"은 잴 수 없고, 4개 중 어느 지점이
최댓값인지만 안다. 따라서 여기서 재는 건 "정점이 300s 이전에 이미 지났는가"라는
조악한 대리 지표이며, 그 결과를 그렇게만 해석해야 한다(과대 해석 금지).

출력: view_count별 "정점이 300s 이전(=조기 정점)"인 궤적의 비율, 그리고 조기 정점인
경우의 하강폭(정점 PSNR - 300s PSNR)을 scene을 독립 단위로 한 cluster bootstrap CI로
낸다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from protocol_utils import scene_cluster_bootstrap_ci  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = REPO_ROOT / "experiments/outputs/re10k_c1a_main"
VIEW_COUNTS = [2, 4, 8, 12]
METHODS = ["Vanilla3DGS", "FSGS"]


def load_trajectories(method: str) -> list[dict]:
    subdir = "vanilla_runs" if method == "Vanilla3DGS" else "fsgs_runs"
    rows = []
    for log_path in sorted((RUN_DIR / subdir).glob("*/logs/*.json")):
        traj = json.loads(log_path.read_text())
        if len(traj) != 4:
            continue  # 예산 스냅샷 4개가 아니면(비정상 로그) 건너뜀
        traj = sorted(traj, key=lambda r: r["wall_clock"])
        scene = traj[0]["scene"]
        seed = traj[0]["seed"]
        # scene 이름에서 view_count 정보가 없으므로 로그 파일명에서 추출
        name = log_path.stem  # 예: re10k_<scene>_c1a_Vanilla3DGS_8view_seed0
        view_count = int(name.split("_")[-2].replace("view", ""))
        peak_idx = max(range(4), key=lambda i: traj[i]["test_psnr"])
        peak_budget = traj[peak_idx]["wall_clock"]
        peak_psnr = traj[peak_idx]["test_psnr"]
        final_psnr = traj[-1]["test_psnr"]
        early_peak = peak_idx < 3  # 300s 이전(1/10/60s 중 하나)에 최댓값
        decay = peak_psnr - final_psnr if early_peak else 0.0
        rows.append({
            "scene": scene, "seed": seed, "view_count": view_count, "method": method,
            "peak_budget": peak_budget, "peak_psnr": peak_psnr, "final_psnr": final_psnr,
            "early_peak": early_peak, "decay": decay,
        })
    return rows


def main() -> int:
    all_rows = []
    for method in METHODS:
        all_rows.extend(load_trajectories(method))

    print(f"loaded {len(all_rows)} trajectories\n")

    for method in METHODS:
        print(f"=== {method} ===")
        print(f"{'view':>5} {'early-peak rate':>16} {'n_traj':>7} | {'decay(early-peak only)':>24} {'n_scene':>8}")
        for vc in VIEW_COUNTS:
            subset = [r for r in all_rows if r["method"] == method and r["view_count"] == vc]
            if not subset:
                continue
            early_rate = mean(1.0 if r["early_peak"] else 0.0 for r in subset)
            early_subset = [r for r in subset if r["early_peak"]]
            if early_subset:
                stat = scene_cluster_bootstrap_ci(early_subset, value_fn=lambda r: r["decay"])
                decay_str = f"{stat['mean']:.2f}dB [{stat['ci_low']:.2f},{stat['ci_high']:.2f}]"
                n_scene = stat["scene_count"]
            else:
                decay_str = "n/a (no early peak)"
                n_scene = 0
            print(f"{vc:>5} {early_rate*100:>15.1f}% {len(subset):>7} | {decay_str:>24} {n_scene:>8}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
