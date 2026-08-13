#!/usr/bin/env python3
"""C1-a 본 실험 — RE10K 30 scene × 2/4/8/12-view × budget[1,10,60,300]s, seed{0,1}.

`run_re10k_c1a_pilot.py`(5 scene, seed 1회, Vanilla3DGS vs MVSplat)를 2026-08-13 grid 결정
(overall.md §5.4: scene 20->30, seed 3->2)과 FSGS 합류(§4.2)를 반영해 확장한 것. 세 방법론을
같은 (scene, view_count, seed) 조건에서 비교한다:
- MVSplat(feed-forward): 단일 추론, deterministic이라 seed 무관 1회만
- Vanilla3DGS(optimization, COLMAP/random init): seed{0,1} 반복
- FSGS(sparse-view 특화 optimization, sparse-init 대체 — overall.md §4.2 명시적 편차): seed{0,1} 반복

**이번 착수 범위(2026-08-13)**: overlap_level(고/저) 축은 포함하지 않는다 — co-visibility
selector(`core/view_selector.py`)가 아직 pilot 3-scene 규모로만 검증됐고(RE10K 12/12,
DL3DV 11/12) 30-scene 전체 재생성이 안 된 상태라, 검증 안 된 축까지 넣어 84 GPU-hour를
한 번에 태우는 위험을 피한다. view_count×budget×method×seed 축만 먼저 완주하고,
overlap_level은 selector를 30-scene으로 확장한 뒤 별도로 추가한다.

summary는 scene 하나가 끝날 때마다 즉시 디스크에 다시 쓴다(끝까지 기다리지 않음) —
장시간(수십 시간) 백그라운드 job이라 중간에 대시보드로 진행 상황을 봐야 한다.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MVSPLAT_PY = "/opt/conda/envs/mvsplat/bin/python3"
PS3_PY = "/opt/conda/envs/ps3/bin/python3"
FSGS_PY = "/opt/conda/envs/fsgs/bin/python3"
BUDGETS = [1.0, 10.0, 60.0, 300.0]
SEEDS = [0, 1]


def _env_with_bin(conda_env_python: str) -> dict:
    conda_bin = str(Path(conda_env_python).parent)
    env = os.environ.copy()
    env["PATH"] = f"{conda_bin}:{env.get('PATH', '')}"
    return env


def run_step(cmd: list[str], label: str) -> bool:
    result = subprocess.run(cmd, capture_output=True, text=True, env=_env_with_bin(cmd[0]))
    ok = result.returncode == 0
    if not ok:
        print(f"  [FAIL] {label}: {result.stderr[-1200:]}")
    return ok


def write_summary(output_dir: Path, rows: list[dict]) -> None:
    summary_path = output_dir / "c1a_main_summary.json"
    tmp_path = summary_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    tmp_path.replace(summary_path)  # atomic — 대시보드가 쓰다 만 파일을 읽지 않게


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-index", default=str(REPO_ROOT / "experiments/outputs/re10k_main_subset/re10k_main_subset.json"))
    parser.add_argument("--scenes", nargs="+", default=None, help="지정 안 하면 subset 전체(30 scene)")
    parser.add_argument("--view-counts", type=int, nargs="+", default=[2, 4, 8, 12])
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--budgets", type=float, nargs="+", default=BUDGETS)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "experiments/outputs/re10k_c1a_main"))
    args = parser.parse_args()

    subset = json.loads(Path(args.subset_index).read_text())
    scene_keys = args.scenes if args.scenes else list(subset.keys())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    t_start = time.time()
    total = len(scene_keys) * len(args.view_counts)
    step = 0

    for scene_key in scene_keys:
        for view_count in args.view_counts:
            step += 1
            candidate = subset[scene_key]["view_candidates"].get(str(view_count), {})
            if candidate.get("context") is None:
                print(f"[skip {step}/{total}] {scene_key} {view_count}view: candidate 없음")
                continue
            print(f"[{step}/{total}] {scene_key} {view_count}view")
            t0 = time.time()

            # --- MVSplat (feed-forward, single deterministic inference) ---
            mv_log = output_dir / "logs" / f"{scene_key}_MVSplat_{view_count}view.json"
            if not mv_log.exists():
                run_step(
                    [
                        MVSPLAT_PY, str(REPO_ROOT / "experiments/scripts/runners/mvsplat_re10k_runner.py"),
                        "--subset-index", args.subset_index, "--scene-key", scene_key,
                        "--view-count", str(view_count), "--output-dir", str(output_dir),
                    ],
                    "mvsplat_re10k_runner",
                )
            if mv_log.exists():
                mv_row = json.loads(mv_log.read_text())
                for budget in args.budgets:
                    rows.append({
                        "scene": scene_key, "view_count": view_count, "budget": budget,
                        "method": "MVSplat", "seed": None,
                        "status": "ok" if mv_row["wall_clock"] <= budget else "no_result_budget_exceeded",
                        "test_psnr": mv_row["test_psnr"] if mv_row["wall_clock"] <= budget else None,
                        "wall_clock": mv_row["wall_clock"],
                    })

            # --- Vanilla3DGS (optimization, COLMAP/random init), seed별 반복 ---
            vanilla_output = output_dir / "vanilla_runs" / scene_key
            for seed in args.seeds:
                log_path = vanilla_output / "logs" / f"re10k_{scene_key}_c1a_Vanilla3DGS_{view_count}view_seed{seed}.json"
                if not log_path.exists():
                    run_step(
                        [
                            PS3_PY, str(REPO_ROOT / "experiments/scripts/runners/vanilla_3dgs_runner.py"),
                            "--dataset", "re10k", "--re10k-scene-key", scene_key,
                            "--re10k-subset-index", args.subset_index,
                            "--scene", f"re10k_{scene_key}_c1a", "--view-count", str(view_count), "--seed", str(seed),
                            "--image-shape", "256", "256",
                            "--max-budget-seconds", str(max(args.budgets)),
                            "--budget-snapshots", *[str(b) for b in args.budgets],
                            "--output-dir", str(vanilla_output),
                        ],
                        "vanilla_3dgs_runner(re10k general)",
                    )
                if log_path.exists():
                    trajectory = json.loads(log_path.read_text())
                    by_budget = {r["wall_clock"]: r for r in trajectory}
                    for budget in args.budgets:
                        r = by_budget.get(budget)
                        rows.append({
                            "scene": scene_key, "view_count": view_count, "budget": budget, "seed": seed,
                            "method": "Vanilla3DGS",
                            "status": "ok" if r else "no_result",
                            "test_psnr": r["test_psnr"] if r else None,
                            "init_source": r["init_source"] if r else None,
                        })

            # --- FSGS (sparse-view 특화 optimization, sparse-init 대체), seed별 반복 ---
            fsgs_output = output_dir / "fsgs_runs" / scene_key
            for seed in args.seeds:
                log_path = fsgs_output / "logs" / f"re10k_{scene_key}_c1a_FSGS_{view_count}view_seed{seed}.json"
                if not log_path.exists():
                    run_step(
                        [
                            FSGS_PY, str(REPO_ROOT / "experiments/scripts/runners/fsgs_runner.py"),
                            "--dataset", "re10k", "--re10k-scene-key", scene_key,
                            "--re10k-subset-index", args.subset_index,
                            "--scene", f"re10k_{scene_key}_c1a", "--view-count", str(view_count), "--seed", str(seed),
                            "--image-shape", "256", "256",
                            "--max-budget-seconds", str(max(args.budgets)),
                            "--budget-snapshots", *[str(b) for b in args.budgets],
                            "--output-dir", str(fsgs_output),
                        ],
                        "fsgs_runner(re10k)",
                    )
                if log_path.exists():
                    trajectory = json.loads(log_path.read_text())
                    by_budget = {r["wall_clock"]: r for r in trajectory}
                    for budget in args.budgets:
                        r = by_budget.get(budget)
                        rows.append({
                            "scene": scene_key, "view_count": view_count, "budget": budget, "seed": seed,
                            "method": "FSGS",
                            "status": "ok" if r else "no_result",
                            "test_psnr": r["test_psnr"] if r else None,
                            "init_source": r["init_source"] if r else None,
                        })

            print(f"  done in {time.time()-t0:.1f}s ({len(rows)} rows so far)")
            write_summary(output_dir, rows)  # view_count 조합 하나 끝날 때마다 갱신 — 대시보드가 이걸 읽는다

    write_summary(output_dir, rows)
    print(f"\n[done] {len(rows)} rows in {(time.time()-t_start)/3600:.2f}h. summary: {output_dir / 'c1a_main_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
