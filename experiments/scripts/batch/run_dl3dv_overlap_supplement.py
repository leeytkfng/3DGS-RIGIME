#!/usr/bin/env python3
"""Overlap 축 보조 실험 — DL3DV 8-scene(seed=0 결정론적 서브샘플) × 2/4/8/12-view ×
high/low overlap × budget[1,10,60,300]s, seed=0 (단일).

`run_re10k_overlap_supplement.py`를 DL3DV로 그대로 이식한 것 — 다른 점은
`run_dl3dv_c1a_main.py`가 RE10K판과 다른 것과 같은 두 가지뿐이다: feed-forward가
MVSplat이 아니라 **DepthSplat**(DL3DV in-domain 체크포인트), 입력 해상도가 256×448.

8-scene은 `dl3dv_overlap_lowhigh/dl3dv_overlap_candidates.json`(25-scene 전체, A-2
진단·C1-a에도 쓴 후보 풀)에서 seed=0으로 결정론적 서브샘플(RE10K와 동일 절차).

이 스크립트를 준비하며 `depthsplat_dl3dv_runner.py`에서 실제 버그를 하나 발견해 고쳤다:
log_path/checkpoints_dir가 overlap_suffix를 안 붙이고 있어서(`mvsplat_re10k_runner.py`는
붙임) high 다음에 low를 돌리면 두 번째 실행이 "이미 로그 있음"으로 오인되어 건너뛰고
high 결과를 low인 것처럼 재사용할 뻔했다 — 이 스크립트를 쓰기 전에 수정 완료
(2026-08-18, `depthsplat_dl3dv_runner.py`).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DEPTHSPLAT_PY = "/opt/conda/envs/depthsplat/bin/python3"
PS3_PY = "/opt/conda/envs/ps3/bin/python3"
FSGS_PY = "/opt/conda/envs/fsgs/bin/python3"
BUDGETS = [1.0, 10.0, 60.0, 300.0]
SEED = 0
NUM_SCENES = 8
IMAGE_SHAPE = ["256", "448"]


def _env_with_bin(conda_env_python: str) -> dict:
    env = os.environ.copy()
    env["PATH"] = f"{Path(conda_env_python).parent}:{env.get('PATH', '')}"
    return env


def run_step(cmd: list[str], label: str) -> bool:
    result = subprocess.run(cmd, capture_output=True, text=True, env=_env_with_bin(cmd[0]))
    ok = result.returncode == 0
    if not ok:
        print(f"  [FAIL] {label}: {result.stderr[-1200:]}")
    return ok


def write_summary(output_dir: Path, rows: list[dict]) -> None:
    summary_path = output_dir / "overlap_supplement_summary.json"
    tmp_path = summary_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    tmp_path.replace(summary_path)


def select_scenes(candidates_path: Path) -> list[str]:
    all_scenes = sorted(json.loads(candidates_path.read_text()).keys())
    rng = np.random.default_rng(SEED)
    return sorted(rng.choice(all_scenes, size=NUM_SCENES, replace=False).tolist())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlap-summary", default=str(REPO_ROOT / "experiments/outputs/dl3dv_overlap_v2/all_scenes_summary.json"),
                         help="DepthSplat --overlap-level 미지정 시 기본 view 선택 풀(사용 안 함, 항상 --overlap-level 지정).")
    parser.add_argument("--overlap-candidates-index", default=str(REPO_ROOT / "experiments/outputs/dl3dv_overlap_lowhigh/dl3dv_overlap_candidates.json"))
    parser.add_argument("--scenes", nargs="+", default=None, help="지정 안 하면 seed=0으로 8-scene 결정론적 선택")
    parser.add_argument("--view-counts", type=int, nargs="+", default=[2, 4, 8, 12])
    parser.add_argument("--levels", nargs="+", default=["high", "low"])
    parser.add_argument("--budgets", type=float, nargs="+", default=BUDGETS)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "experiments/outputs/dl3dv_overlap_supplement"))
    args = parser.parse_args()

    scene_keys = args.scenes or select_scenes(Path(args.overlap_candidates_index))
    print(f"[scenes] {len(scene_keys)}: {scene_keys}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    t_start = time.time()
    total = len(scene_keys) * len(args.view_counts) * len(args.levels)
    step = 0

    for scene_key in scene_keys:
        for view_count in args.view_counts:
            for level in args.levels:
                step += 1
                print(f"[{step}/{total}] {scene_key} {view_count}view {level}")
                t0 = time.time()

                # --- DepthSplat (feed-forward, single deterministic inference, DL3DV in-domain) ---
                ds_log = output_dir / "logs" / f"{scene_key}_DepthSplat_{view_count}view_{level}.json"
                if not ds_log.exists():
                    run_step(
                        [
                            DEPTHSPLAT_PY, str(REPO_ROOT / "experiments/scripts/runners/depthsplat_dl3dv_runner.py"),
                            "--overlap-summary", args.overlap_summary, "--scene-key", scene_key,
                            "--overlap-level", level, "--overlap-candidates-index", args.overlap_candidates_index,
                            "--view-count", str(view_count), "--output-dir", str(output_dir),
                        ],
                        "depthsplat_dl3dv_runner",
                    )
                if ds_log.exists():
                    ds_row = json.loads(ds_log.read_text())
                    for budget in args.budgets:
                        rows.append({
                            "scene": scene_key, "view_count": view_count, "overlap_level": level, "budget": budget,
                            "method": "DepthSplat", "seed": None,
                            "status": "ok" if ds_row["wall_clock"] <= budget else "no_result_budget_exceeded",
                            "test_psnr": ds_row["test_psnr"] if ds_row["wall_clock"] <= budget else None,
                        })

                # --- Vanilla3DGS ---
                vanilla_output = output_dir / "vanilla_runs" / scene_key
                log_path = vanilla_output / "logs" / f"dl3dv_{scene_key}_ovl_Vanilla3DGS_{view_count}view_seed{SEED}_{level}.json"
                if not log_path.exists():
                    run_step(
                        [
                            PS3_PY, str(REPO_ROOT / "experiments/scripts/runners/vanilla_3dgs_runner.py"),
                            "--dataset", "dl3dv", "--dl3dv-scene-key", scene_key,
                            "--overlap-level", level, "--dl3dv-overlap-candidates-index", args.overlap_candidates_index,
                            "--scene", f"dl3dv_{scene_key}_ovl", "--view-count", str(view_count), "--seed", str(SEED),
                            "--image-shape", *IMAGE_SHAPE,
                            "--max-budget-seconds", str(max(args.budgets)),
                            "--budget-snapshots", *[str(b) for b in args.budgets],
                            "--output-dir", str(vanilla_output),
                        ],
                        "vanilla_3dgs_runner(dl3dv overlap)",
                    )
                if log_path.exists():
                    trajectory = json.loads(log_path.read_text())
                    by_budget = {r["wall_clock"]: r for r in trajectory}
                    for budget in args.budgets:
                        r = by_budget.get(budget)
                        rows.append({
                            "scene": scene_key, "view_count": view_count, "overlap_level": level, "budget": budget,
                            "seed": SEED, "method": "Vanilla3DGS",
                            "status": "ok" if r else "no_result",
                            "test_psnr": r["test_psnr"] if r else None,
                        })

                # --- FSGS ---
                fsgs_output = output_dir / "fsgs_runs" / scene_key
                log_path = fsgs_output / "logs" / f"dl3dv_{scene_key}_ovl_FSGS_{view_count}view_seed{SEED}_{level}.json"
                if not log_path.exists():
                    run_step(
                        [
                            FSGS_PY, str(REPO_ROOT / "experiments/scripts/runners/fsgs_runner.py"),
                            "--dataset", "dl3dv", "--dl3dv-scene-key", scene_key,
                            "--overlap-level", level, "--dl3dv-overlap-candidates-index", args.overlap_candidates_index,
                            "--scene", f"dl3dv_{scene_key}_ovl", "--view-count", str(view_count), "--seed", str(SEED),
                            "--image-shape", *IMAGE_SHAPE,
                            "--max-budget-seconds", str(max(args.budgets)),
                            "--budget-snapshots", *[str(b) for b in args.budgets],
                            "--output-dir", str(fsgs_output),
                        ],
                        "fsgs_runner(dl3dv overlap)",
                    )
                if log_path.exists():
                    trajectory = json.loads(log_path.read_text())
                    by_budget = {r["wall_clock"]: r for r in trajectory}
                    for budget in args.budgets:
                        r = by_budget.get(budget)
                        rows.append({
                            "scene": scene_key, "view_count": view_count, "overlap_level": level, "budget": budget,
                            "seed": SEED, "method": "FSGS",
                            "status": "ok" if r else "no_result",
                            "test_psnr": r["test_psnr"] if r else None,
                        })

                print(f"  done in {time.time()-t0:.1f}s ({len(rows)} rows so far)")
                write_summary(output_dir, rows)

    write_summary(output_dir, rows)
    print(f"\n[done] {len(rows)} rows in {(time.time()-t_start)/3600:.2f}h. summary: {output_dir / 'overlap_supplement_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
