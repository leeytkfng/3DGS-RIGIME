#!/usr/bin/env python3
"""DTU scan을 FSGS(`scene/dataset_readers.py::readColmapSceneInfo`)가 기대하는 디렉토리
구조로 준비한다 — 첫 실제 학습 검증용, 아직 protocol_utils 러너는 아니다.

FSGS는 원래 `colmap patch_match_stereo`로 만든 dense point cloud(`{n_views}_views/dense/
fused.ply`)를 기대하는데, 우리 시스템엔 COLMAP CLI(dense MVS)가 없다(pycolmap 파이썬
바인딩만 있음, `patch_match_stereo`는 여기 없음). 대신 우리가 이미 갖고 있는 sparse
triangulation 결과(pycolmap known-pose triangulation, 다른 러너들과 동일한 코어)를
그 자리에 넣는다 — dense가 아니라 sparse 초기화라는 차이는 있지만, Vanilla3DGS도 원래
sparse COLMAP 초기화를 쓰므로 방법론적으로 이상하지 않다.

만드는 구조:
  <out>/images/*.png              (49 view 전체)
  <out>/sparse/0/{cameras.bin,images.bin}   (49 view 전체, pose 등록용)
  <out>/{n_views}_views/dense/fused.ply     (train pool만으로 triangulate한 sparse point)
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pycolmap

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from colmap_init import CameraParams, _write_known_pose_reconstruction_generic, triangulate_sfm_points_from_cameras  # noqa: E402
from dtu_dataset import load_camera  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-dir", default="/data/Re-feem/datasets/dtu/scan1")
    parser.add_argument("--n-views", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="experiments/outputs/fsgs_data/dtu_scan1")
    args = parser.parse_args()

    scan_dir = Path(args.scan_dir)
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)

    # 다른 러너들과 동일한 held-out 규칙: 1,8,15,...,43이 test, 나머지가 train pool.
    all_view_ids = list(range(1, 50))
    test_ids = all_view_ids[::7]
    train_pool = [v for v in all_view_ids if v not in test_ids]
    rng = np.random.default_rng(args.seed)
    train_ids = sorted(rng.choice(train_pool, size=min(args.n_views, len(train_pool)), replace=False).tolist())
    print(f"[data] train views ({len(train_ids)}): {train_ids}")
    print(f"[data] test views ({len(test_ids)}): {test_ids}")

    # 1) images/ 전체 복사 (49 view 전부 -- FSGS의 llffhold eval-split이 전체 pose set을 기대함)
    images_out = out_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)
    for v in all_view_ids:
        shutil.copy(scan_dir / "images" / f"{v:03d}.png", images_out / f"{v:03d}.png")

    # 2) sparse/0/{cameras.bin,images.bin}: 49 view 전체 pose를 known-pose로 등록.
    #    실제 feature matching은 필요 없다(포즈가 이미 DTU calibration으로 알려져 있음) —
    #    전체 COLMAP triangulation 파이프라인(추출+매칭)을 다 돌리는 건 낭비이므로, pose만
    #    기록하는 내부 헬퍼(`_write_known_pose_reconstruction_generic`)를 직접 재사용한다.
    calibration_dir = scan_dir / "cameras"
    all_image_names = [f"{v:03d}.png" for v in all_view_ids]
    camera_by_name_all = {}
    for v, name in zip(all_view_ids, all_image_names):
        cam = load_camera(calibration_dir, v)
        camera_by_name_all[name] = CameraParams(K=cam.K, R=cam.R, t=cam.t, width=1600, height=1200)

    sparse_dir = out_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    id_map = {name: (idx + 1, idx + 1) for idx, name in enumerate(all_image_names)}
    _write_known_pose_reconstruction_generic(sparse_dir, camera_by_name_all, id_map)
    # readColmapSceneInfo는 .bin을 우선 시도하고 실패하면 .txt로 fallback하므로 .txt만 있어도 된다.

    # FSGS의 readColmapCameras()는 LLFF 관례의 poses_bounds.npy(근/원 깊이 경계)도 요구한다.
    # idx는 정렬된 image name 순서(=all_image_names와 동일 순서)와 일치해야 한다. 실제 pose는
    # COLMAP에서 오므로 앞 15열(3x5 pose)은 안 쓰이고 마지막 2열(near, far)만 읽힌다 — MVSplat
    # DTU config(`config/experiment/dtu.yaml`)의 near=2.125/far=4.525를 그대로 재사용.
    poses_bounds = np.zeros((len(all_image_names), 17), dtype=np.float64)
    poses_bounds[:, -2] = 2.125
    poses_bounds[:, -1] = 4.525
    np.save(out_dir / "poses_bounds.npy", poses_bounds)

    # 3) {n_views}_views/dense/fused.ply: train pool(=희소 view만)로 triangulate한 점을
    #    dense 대신 채워넣는다 (§docstring 참고).
    train_image_names = [f"{v:03d}.png" for v in train_ids]
    camera_by_name_train = {name: camera_by_name_all[name] for name in train_image_names}
    sparse_workdir = out_dir / "_sparse_triangulation"
    points, colors = triangulate_sfm_points_from_cameras(images_out, train_image_names, camera_by_name_train, sparse_workdir)
    print(f"[colmap] sparse triangulation: {points.shape[0]} points from {len(train_ids)} train views")

    dense_dir = out_dir / f"{args.n_views}_views" / "dense"
    dense_dir.mkdir(parents=True, exist_ok=True)
    _write_ply(dense_dir / "fused.ply", points, colors)
    print(f"[done] FSGS-ready data at {out_dir}")
    print(f"  images: {len(all_view_ids)}, pose-registered: {len(all_view_ids)}, sparse points: {points.shape[0]}")
    return 0


def _write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    from plyfile import PlyData, PlyElement

    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"),
             ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
             ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate([xyz, normals, rgb], axis=1)
    elements[:] = list(map(tuple, attributes))
    PlyData([PlyElement.describe(elements, "vertex")]).write(str(path))


if __name__ == "__main__":
    raise SystemExit(main())
