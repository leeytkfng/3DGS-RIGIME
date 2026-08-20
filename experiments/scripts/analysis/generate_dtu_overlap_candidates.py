#!/usr/bin/env python3
"""DTU에 co-visibility selector(`core/view_selector.py`)를 적용해 C2의
`representative_conditions`(2view_low_overlap / 4view_low_overlap / 4view_high_overlap /
12view_high_overlap)에 필요한 high/low overlap context 후보를 만든다.

RE10K/DL3DV용 `generate_re10k_overlap_candidates.py`/`generate_dl3dv_overlap_candidates.py`와
같은 목적/패턴. DTU만의 차이:

- **test(target) view는 이 프로젝트의 기존 DTU 고정 held-out split을 그대로 쓴다**
  (`vanilla_3dgs_runner.py`/`precompute_depth_maps.py`의 `all_view_ids[::7]`,
  view_id 1,8,15,...,43 — C1-b gate 등 기존 DTU 작업 전체가 이미 이 split을 쓰고 있어서
  바꾸지 않는다). high/low 사이에도 target을 고정해 DL3DV처럼 "high/low가 target까지
  달라지는" 추가 교란을 만들지 않는다(A-2 진단에서 확인한 문제를 여기서는 미리 피한다).
- context(train) pool은 그 나머지 42개 view.

**8개 scene 선택**: DTU 전체(29 scan) 중에서, 이 프로젝트가 이미 C1-b 등에서 계속 써온
MVSplat 공식 sparse-view test split(`OFFICIAL_DTU_SPLIT`, 16개, `run_dtu_batch.py` 참고)
안에서, `np.random.default_rng(0)`으로 8개를 뽑는다 — 결과를 보고 고르지 않도록 파일럿 전에
결정론적으로 고정(§5.9 "파일럿 전에 대표 조건을 고정" 원칙과 동일하게, scene 선택에도 적용).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pycolmap

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from colmap_init import CameraParams, triangulate_sfm_points_from_cameras  # noqa: E402
from dtu_dataset import load_camera, load_image  # noqa: E402
from protocol_utils import build_overlap_report  # noqa: E402
from view_selector import select_overlap_candidates  # noqa: E402

DTU_ROOT = Path("/data/Re-feem/datasets/dtu")

# MVSplat 공식 sparse-view test split(§`run_dtu_batch.py` OFFICIAL_DTU_SPLIT) — 이 프로젝트가
# DTU 작업 전체에서 계속 재사용해온 16-scan 풀. C2는 이 안에서 8개만 쓴다.
OFFICIAL_DTU_SPLIT = [1, 8, 21, 30, 31, 34, 38, 40, 41, 45, 55, 63, 82, 103, 110, 114]
C2_SCENE_SEED = 0
C2_NUM_SCENES = 8

VIEW_COUNTS = [2, 4, 12]  # representative_conditions에 실제로 등장하는 view_count만.
SEED = 0  # view_selector FPS seed — RE10K/DL3DV 생성 스크립트와 동일 관례.

# view_selector.py 기본 window_multiplier=4.0은 RE10K/DL3DV(pool 수백~수천)를 염두에 둔 값.
# DTU는 held-out 7개를 뺀 context pool이 42개뿐이라, view_count=12일 때 기본값을 쓰면
# window_size=min(42, 12*4=48)=42=pool 전체가 돼버려 "high"가 "low"와 사실상 같아진다
# (2026-08-16 첫 실행 실측: 12-view 방향검증 8scan 중 5개 FAIL, high==low 수치까지 나옴).
# 42라는 pool 크기에서 최대 view_count(12)에도 진짜 좁은 창이 되도록 2.0으로 낮춘다
# (window_size: 2view=4, 4view=8, 12view=24 — 전부 42보다 확실히 작음).
DTU_WINDOW_MULTIPLIER = 2.0


def select_c2_scenes() -> list[int]:
    rng = np.random.default_rng(C2_SCENE_SEED)
    return sorted(rng.choice(OFFICIAL_DTU_SPLIT, size=C2_NUM_SCENES, replace=False).tolist())


def dtu_camera_positions(view_ids: list[int], calibration_dir: Path) -> np.ndarray:
    return np.stack([load_camera(calibration_dir, v).center for v in view_ids])


def measure_overlap(scan_dir: Path, context_ids: list[int], workdir: Path) -> dict:
    image_dir = scan_dir / "images"
    calibration_dir = scan_dir / "cameras"
    image_names = [f"{v:03d}.png" for v in context_ids]

    camera_by_name = {}
    for v, name in zip(context_ids, image_names):
        camera = load_camera(calibration_dir, v)
        height, width = load_image(image_dir, v).shape[:2]
        camera_by_name[name] = CameraParams(K=camera.K, R=camera.R, t=camera.t, width=width, height=height)

    points, _ = triangulate_sfm_points_from_cameras(image_dir, image_names, camera_by_name, workdir)
    model_dir = workdir / "sparse_triangulated"
    if points.shape[0] > 0 and (model_dir / "images.bin").exists():
        reconstruction = pycolmap.Reconstruction(str(model_dir))
        visibility = {
            image.name: {str(int(p.point3D_id)) for p in image.points2D if p.point3D_id != -1}
            for image in reconstruction.images.values()
        }
    else:
        visibility = {name: set() for name in image_names}
    report = build_overlap_report(visibility, min_common_points=1)
    return {"sfm_points": int(points.shape[0]), **report["summary"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scans", type=int, nargs="+", default=None, help="지정 안 하면 select_c2_scenes()로 8개 자동 선택.")
    parser.add_argument("--view-counts", type=int, nargs="+", default=VIEW_COUNTS)
    parser.add_argument("--output", default="experiments/outputs/dtu_overlap_candidates/dtu_overlap_candidates.json")
    parser.add_argument("--workdir", default="experiments/outputs/dtu_overlap_candidates/colmap_work")
    args = parser.parse_args()

    scans = args.scans or select_c2_scenes()
    print(f"[scenes] DTU C2 8-scene set (seed={C2_SCENE_SEED} from OFFICIAL_DTU_SPLIT): {scans}")

    all_view_ids = list(range(1, 50))
    test_ids = sorted(all_view_ids[::7])  # 기존 DTU 고정 held-out split과 동일.
    train_pool = [v for v in all_view_ids if v not in test_ids]

    output: dict[str, dict] = {}
    for scan_id in scans:
        scene_key = f"scan{scan_id}"
        scan_dir = DTU_ROOT / scene_key
        calibration_dir = scan_dir / "cameras"
        positions = dtu_camera_positions(train_pool, calibration_dir)

        output[scene_key] = {}
        for view_count in args.view_counts:
            candidates = select_overlap_candidates(
                train_pool, positions, view_count, seed=SEED, window_multiplier=DTU_WINDOW_MULTIPLIER
            )
            row = {"target": test_ids}
            for level, context in candidates.items():
                workdir = Path(args.workdir) / scene_key / f"{view_count}view_{level}"
                measured = measure_overlap(scan_dir, context, workdir)
                row[level] = {"context": context, **measured}
                print(
                    f"[{scene_key}][{view_count}view][{level}] context={context} "
                    f"mean_overlap={measured['mean_overlap']:.4f} sfm_points={measured['sfm_points']}"
                )
            output[scene_key][str(view_count)] = row

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"[done] {out_path}")

    print("\n=== high vs low overlap 방향 검증 ===")
    n_correct, n_total = 0, 0
    for scene_key, by_view in output.items():
        for view_count, row in by_view.items():
            if "high" not in row or "low" not in row:
                continue
            n_total += 1
            ok = row["high"]["mean_overlap"] >= row["low"]["mean_overlap"]
            n_correct += int(ok)
            print(
                f"{scene_key} {view_count}view: high={row['high']['mean_overlap']:.4f} "
                f"low={row['low']['mean_overlap']:.4f} {'OK' if ok else 'FAIL(방향 반대)'}"
            )
    print(f"\n{n_correct}/{n_total} 조건에서 high >= low 확인")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
