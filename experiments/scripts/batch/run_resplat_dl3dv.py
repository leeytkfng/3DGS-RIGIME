#!/usr/bin/env python3
"""ReSplat 탐색적 확장 비교 — DL3DV 25-scene x view_count[2,4,8,12].

`resplat_dl3dv_runner.py`를 scene x view_count 그리드로 순회한다. ReSplat은 결정론적
단일 추론(seed 없음, wall_clock~1.5s/run)이라 C1-a처럼 seed x budget 반복이 필요 없다 —
전체 100 콤보가 몇 분 안에 끝난다(2026-08-16 스모크 테스트 기준).

view_count가 체크포인트 학습 view 수(8 또는 16)에서 멀수록 분포 밖(OOD)이다 — 8-view
체크포인트를 기본으로 쓰되, 12-view는 16-view 체크포인트도 같이 돌려서 어느 쪽이 그 조건에
더 맞는지 비교할 수 있게 한다. 결과 해석 시 §5.2 기준으로 OOD를 명시해야 한다.

메인 regime map에는 안 들어간다 — 별도 절의 탐색적 확장 비교 전용(main.tex Limitations 참고).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RESPLAT_PY = "/opt/conda/envs/resplat/bin/python3"
CHECKPOINT_8V = "/data/Re-feem/code/resplat/pretrained/resplat-base-dl3dv-256x448-view8-1934a04c.pth"
CHECKPOINT_16V = "/data/Re-feem/code/resplat/pretrained/resplat-base-dl3dv-256x448-view16-f38bf984.pth"


def _env_with_bin(conda_env_python: str) -> dict:
    conda_bin = str(Path(conda_env_python).parent)
    env = os.environ.copy()
    env["PATH"] = f"{conda_bin}:{env.get('PATH', '')}"
    return env


def run_step(cmd: list[str], label: str) -> bool:
    result = subprocess.run(cmd, capture_output=True, text=True, env=_env_with_bin(cmd[0]))
    ok = result.returncode == 0
    if not ok:
        print(f"  [FAIL] {label}: {result.stderr[-1500:]}")
    return ok


def write_summary(output_dir: Path, rows: list[dict]) -> None:
    summary_path = output_dir / "resplat_dl3dv_summary.json"
    tmp_path = summary_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    tmp_path.replace(summary_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlap-summary", default=str(REPO_ROOT / "experiments/outputs/dl3dv_overlap_v2/all_scenes_summary.json"))
    parser.add_argument("--scenes", nargs="+", default=None, help="지정 안 하면 overlap-summary에 있는 25 scene 전체")
    parser.add_argument("--view-counts", type=int, nargs="+", default=[2, 4, 8, 12])
    parser.add_argument("--num-refine", type=int, default=4)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "experiments/outputs/resplat_dl3dv"))
    args = parser.parse_args()

    overlap_rows = json.loads(Path(args.overlap_summary).read_text())
    available = {(r["scene"], r["view_count"]) for r in overlap_rows if r.get("context_indices") is not None}
    scene_keys = args.scenes if args.scenes else sorted({r["scene"] for r in overlap_rows})
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    t_start = time.time()
    total = len(scene_keys) * len(args.view_counts)
    step = 0

    for scene_key in scene_keys:
        for view_count in args.view_counts:
            step += 1
            if (scene_key, view_count) not in available:
                print(f"[skip {step}/{total}] {scene_key} {view_count}view: candidate 없음")
                continue

            # view_count>=12일 때는 16-view 체크포인트도 같이 (OOD 정도 비교용).
            checkpoints = [("8v", CHECKPOINT_8V)]
            if view_count >= 12:
                checkpoints.append(("16v", CHECKPOINT_16V))

            for ckpt_label, ckpt_path in checkpoints:
                log_path = output_dir / "logs" / f"{scene_key}_ReSplat_{ckpt_label}_{view_count}view.json"
                if not log_path.exists():
                    run_step(
                        [
                            RESPLAT_PY, str(REPO_ROOT / "experiments/scripts/runners/resplat_dl3dv_runner.py"),
                            "--overlap-summary", args.overlap_summary, "--scene-key", scene_key,
                            "--view-count", str(view_count), "--num-refine", str(args.num_refine),
                            "--checkpoint", ckpt_path,
                            "--experiment-id", f"resplat-exploratory-{ckpt_label}",
                            "--output-dir", str(output_dir),
                        ],
                        f"resplat_dl3dv_runner({ckpt_label})",
                    )
                    # 러너의 log 파일명은 checkpoint 라벨을 모르므로, 직접 rename해 구분한다.
                    default_log = output_dir / "logs" / f"{scene_key}_ReSplat_{view_count}view.json"
                    if default_log.exists() and not log_path.exists():
                        default_log.rename(log_path)

                if log_path.exists():
                    row = json.loads(log_path.read_text())
                    rows.append({**row, "view_count": view_count, "checkpoint_label": ckpt_label})

            print(f"[{step}/{total}] {scene_key} {view_count}view done ({len(rows)} rows so far)")
            write_summary(output_dir, rows)

    write_summary(output_dir, rows)
    print(f"\n[done] {len(rows)} rows in {(time.time()-t_start)/60:.1f}min. summary: {output_dir / 'resplat_dl3dv_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
