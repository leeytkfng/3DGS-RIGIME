#!/usr/bin/env python3
"""재실행 0 분석 — 부록 B(Oracle peak) 채우기.

이미 저장된 C1-a trajectory 로그(`{re10k,dl3dv}_c1a_main/{vanilla,fsgs}_runs/*/logs/*.json`)
만 읽는다. §III.8(계산 예산 시점의 결과 선택 규칙)이 명시한 대로 메인 비교는 예산 종료
시점(budget=300s) 결과만 쓰고 test leakage인 사후 최고점 선택은 하지 않는데, 그 최고점과
예산 종료 시점의 격차가 실제로 얼마나 되는지를 참고용으로만 별도 산출한다(부록 B 원칙).

산출: (dataset, method, view_count)별로
  - oracle_peak_psnr - budget_end(300s)_psnr 의 scene cluster bootstrap 95% CI
  - 정점이 300s 이전(=조기 정점)에 나온 비율(§V.3 H2 조기 정점 서술과 같은 계열의 수치,
    단 여기서는 "그 정점이 얼마나 더 높았는가"의 크기를 본다는 점이 다르다)
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from protocol_utils import scene_cluster_bootstrap_ci  # noqa: E402

TRAJ_NAME_RE = re.compile(r"_(Vanilla3DGS|FSGS)_(\d+)view_seed(\d+)\.json$")


def load_trajectories(root: Path, subdir: str) -> list[dict]:
    """scene_dir/logs/*.json 하나당 (scene, method, view_count, seed) trajectory 한 개."""

    out = []
    for scene_dir in (root / subdir).glob("*"):
        for path in (scene_dir / "logs").glob("*.json"):
            m = TRAJ_NAME_RE.search(path.name)
            if not m:  # densoff/overlap 변형 등은 제외 — 메인 C1-a 궤적만
                continue
            method, view_count, seed = m.group(1), int(m.group(2)), int(m.group(3))
            trajectory = json.loads(path.read_text())
            if not trajectory:
                continue
            budget_end = max(trajectory, key=lambda r: r["wall_clock"])
            peak = max(trajectory, key=lambda r: r["test_psnr"])
            out.append(
                {
                    "scene": scene_dir.name,
                    "method": method,
                    "view_count": view_count,
                    "seed": seed,
                    "budget_end_wall_clock": budget_end["wall_clock"],
                    "budget_end_psnr": budget_end["test_psnr"],
                    "peak_wall_clock": peak["wall_clock"],
                    "peak_psnr": peak["test_psnr"],
                    "gap": peak["test_psnr"] - budget_end["test_psnr"],
                    "early_peak": peak["wall_clock"] < budget_end["wall_clock"],
                }
            )
    return out


def average_seeds(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[(r["scene"], r["method"], r["view_count"])].append(r)
    out = []
    for (scene, method, vc), group in grouped.items():
        out.append(
            {
                "scene": scene, "method": method, "view_count": vc,
                "gap": float(np.mean([g["gap"] for g in group])),
                "early_peak_rate": float(np.mean([g["early_peak"] for g in group])),
            }
        )
    return out


def analyze(dataset: str) -> list[dict]:
    root = Path(f"experiments/outputs/{dataset.lower()}_c1a_main")
    raw = load_trajectories(root, "vanilla_runs") + load_trajectories(root, "fsgs_runs")
    seed_avg = average_seeds(raw)

    print(f"\n=== {dataset}: Oracle peak gap (peak PSNR - budget=300s PSNR), scene cluster bootstrap 95% CI ===")
    summary_rows = []
    by_method_vc: dict[tuple, list[dict]] = defaultdict(list)
    for r in seed_avg:
        by_method_vc[(r["method"], r["view_count"])].append(r)

    for (method, vc), group in sorted(by_method_vc.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        ci = scene_cluster_bootstrap_ci(group, lambda r: r["gap"])
        early_rate = float(np.mean([g["early_peak_rate"] for g in group]))
        print(
            f"  [{vc:2d}view][{method:12s}] gap={ci['mean']:+.2f}dB  "
            f"95% CI=[{ci['ci_low']:+.2f}, {ci['ci_high']:+.2f}]  "
            f"early_peak_rate={early_rate:.1%}  (n_scenes={ci['scene_count']})"
        )
        summary_rows.append(
            {
                "dataset": dataset, "method": method, "view_count": vc,
                "gap_mean": ci["mean"], "gap_ci_low": ci["ci_low"], "gap_ci_high": ci["ci_high"],
                "early_peak_rate": early_rate, "n_scenes": ci["scene_count"],
            }
        )
    return summary_rows


def main() -> int:
    all_rows = []
    for dataset in ("RE10K", "DL3DV"):
        all_rows += analyze(dataset)

    out_path = Path("experiments/outputs/oracle_peak_gap/summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    print(f"\n[done] {out_path} ({len(all_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
