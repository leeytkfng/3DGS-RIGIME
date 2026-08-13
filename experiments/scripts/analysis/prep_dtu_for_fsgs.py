#!/usr/bin/env python3
"""FSGS(`scene/dataset_readers.py::readColmapSceneInfo`)가 기대하는 디렉토리 구조로 데이터를
준비한다 — DTU 전용 `prepare_dtu_for_fsgs()`와 dataset-agnostic `prepare_views_for_fsgs()`
(RE10K/DL3DV, 2026-08-13 추가) 둘 다 제공한다. `runners/fsgs_runner.py`가 데이터 prep 단계로
직접 import해서 재사용한다(DTU는 CLI로 단독 실행도 가능, 아래 main() 참고).

FSGS는 원래 `colmap patch_match_stereo`로 만든 dense point cloud(`{n_views}_views/dense/
fused.ply`)를 기대하는데, 우리 시스템엔 COLMAP CLI(dense MVS)가 없다(pycolmap 파이썬
바인딩만 있음, `patch_match_stereo`는 여기 없음). 대신 우리가 이미 갖고 있는 sparse
triangulation 결과(pycolmap known-pose triangulation, 다른 러너들과 동일한 코어)를
그 자리에 넣는다 — dense가 아니라 sparse 초기화라는 차이는 있지만, Vanilla3DGS도 원래
sparse COLMAP 초기화를 쓰므로 방법론적으로 이상하지 않다.

만드는 구조(공통):
  <out>/images/*.png              (등록된 모든 view)
  <out>/sparse/0/{cameras.bin,images.bin}   (pose 등록용)
  <out>/{n_views}_views/dense/fused.ply     (train view만으로 triangulate한 sparse point)
  <out>/poses_bounds.npy          (LLFF 관례, near/far만 실제로 읽힘)
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pycolmap
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from colmap_init import CameraParams, _write_known_pose_reconstruction_generic, triangulate_sfm_points_from_cameras  # noqa: E402
from dtu_dataset import estimate_scene_sphere, load_camera  # noqa: E402

MIN_SFM_POINTS = 200  # vanilla_3dgs_runner.py와 동일 기준 — 이보다 적으면 random-sphere fallback


def _random_points_in_sphere(center: np.ndarray, radius: float, num_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """vanilla_3dgs_runner.py의 동명 함수와 동일한 로직 — 저 overlap(예: 2-view)에서
    triangulation이 거의/전혀 안 나올 때 빈 Gaussian(rasterizer가 CUDA 오류로 죽음, 2026-08-13
    실측 확인)을 막는다."""

    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(num_points, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = radius * (rng.uniform(size=(num_points, 1)) ** (1.0 / 3.0))
    points = center[None, :] + directions * radii
    colors = np.full((num_points, 3), 128, dtype=np.uint8)
    return points, colors


def prepare_views_for_fsgs(
    train_views: list[dict],
    test_views: list[dict],
    seed: int,
    out_dir: Path,
    near: float,
    far: float,
    num_fallback_points: int = 100_000,
) -> tuple[list[str], list[str], str]:
    """RE10K/DL3DV처럼 이미 메모리에 로드된 view(image/K/R/t/width/height/center)로 FSGS-ready
    디렉토리를 만든다. `prepare_dtu_for_fsgs()`의 dataset-agnostic 버전 — DTU는 view id가
    1..49 정수라 직접 다뤘지만, RE10K/DL3DV는 원본 프레임 인덱스가 정돈된 정수가 아니므로
    image_name(str)을 우리가 직접 붙여서 반환한다(`vanilla_3dgs_runner.py`의
    `_colmap_init_from_loaded_views()`와 같은 패턴 — 이미 로드된 view를 임시 디렉토리에
    써서 known-pose triangulation 코어를 재사용).

    n_views(=FSGS `--n_views`, `{n_views}_views/dense/fused.ply` 경로명)는 len(train_views)로
    고정한다 — FSGS의 `readColmapSceneInfo()`가 `assert len(train_cam_infos) == n_views`로
    검증하므로 반드시 일치해야 한다.
    """

    if out_dir.exists():
        shutil.rmtree(out_dir)

    n_views = len(train_views)
    train_names = [f"train_{i:04d}.png" for i in range(len(train_views))]
    test_names = [f"test_{i:04d}.png" for i in range(len(test_views))]

    images_out = out_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)
    camera_by_name: dict[str, CameraParams] = {}
    for name, view in zip(train_names + test_names, train_views + test_views):
        image_uint8 = (np.clip(view["image"], 0.0, 1.0) * 255.0).astype(np.uint8)
        Image.fromarray(image_uint8).save(images_out / name)
        camera_by_name[name] = CameraParams(K=view["K"], R=view["R"], t=view["t"], width=view["width"], height=view["height"])

    sparse_dir = out_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    all_names = train_names + test_names
    id_map = {name: (idx + 1, idx + 1) for idx, name in enumerate(all_names)}
    _write_known_pose_reconstruction_generic(sparse_dir, camera_by_name, id_map)

    poses_bounds = np.zeros((len(all_names), 17), dtype=np.float64)
    poses_bounds[:, -2] = near
    poses_bounds[:, -1] = far
    np.save(out_dir / "poses_bounds.npy", poses_bounds)

    camera_by_name_train = {name: camera_by_name[name] for name in train_names}
    sparse_workdir = out_dir / "_sparse_triangulation"
    points, colors = triangulate_sfm_points_from_cameras(images_out, train_names, camera_by_name_train, sparse_workdir)
    print(f"[colmap] sparse triangulation: {points.shape[0]} points from {len(train_names)} train views")

    init_source = "colmap_sfm"
    if points.shape[0] < MIN_SFM_POINTS:
        centers = np.stack([v["center"] for v in train_views])
        center = centers.mean(axis=0)
        radius = float(np.median(np.linalg.norm(centers - center, axis=1))) or 1.0
        points, colors = _random_points_in_sphere(center, radius, num_fallback_points, seed)
        init_source = "random_sphere_fallback"
        print(f"[colmap] fallback: random_sphere_fallback, {points.shape[0]} points (center={center}, radius={radius:.3f})")

    dense_dir = out_dir / f"{n_views}_views" / "dense"
    dense_dir.mkdir(parents=True, exist_ok=True)
    _write_ply(dense_dir / "fused.ply", points, colors)
    print(f"[done] FSGS-ready data at {out_dir}")
    print(f"  images: {len(all_names)}, pose-registered: {len(all_names)}, sparse points: {points.shape[0]}")
    return train_names, test_names, init_source


def prepare_dtu_for_fsgs(
    scan_dir: Path, n_views: int, seed: int, out_dir: Path, num_fallback_points: int = 100_000
) -> tuple[list[int], list[int], str]:
    """DTU scan을 FSGS-ready 디렉토리로 준비하고 (train_ids, test_ids)를 반환한다.

    `fsgs_runner.py`가 이 train_ids/test_ids를 그대로 FSGS의 view-selection monkeypatch에
    넘겨써야 다른 러너(Vanilla3DGS/MVSplat)와 같은 view로 FSGS를 학습시킬 수 있다 —
    FSGS 자체의 `readColmapSceneInfo()`는 llffhold+linspace로 view를 자체 재선정하므로
    이 함수가 만든 train_ids를 모르면 무시된다(§`fsgs_runner.py` docstring 참고).
    """

    if out_dir.exists():
        shutil.rmtree(out_dir)

    # 다른 러너들과 동일한 held-out 규칙: 1,8,15,...,43이 test, 나머지가 train pool.
    all_view_ids = list(range(1, 50))
    test_ids = all_view_ids[::7]
    train_pool = [v for v in all_view_ids if v not in test_ids]
    rng = np.random.default_rng(seed)
    train_ids = sorted(rng.choice(train_pool, size=min(n_views, len(train_pool)), replace=False).tolist())
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

    init_source = "colmap_sfm_sparse"
    if points.shape[0] < MIN_SFM_POINTS:
        # 2-view 등 저 overlap 조건에서 bundle adjustment가 gauge를 못 고정해 0~소수의 점만
        # 나오는 경우가 실제로 있다(2026-08-13, DTU scan1 2-view seed0에서 0점 확인 — 빈
        # Gaussian으로 FSGS의 CUDA rasterizer가 즉시 죽었다). Vanilla3DGS와 동일하게
        # scene bounding sphere 안의 random point로 대체한다.
        cameras_for_sphere = [load_camera(calibration_dir, v) for v in all_view_ids]
        center, radius = estimate_scene_sphere(cameras_for_sphere)
        points, colors = _random_points_in_sphere(center, radius, num_fallback_points, seed)
        init_source = "random_sphere_fallback"
        print(f"[colmap] fallback: random_sphere_fallback, {points.shape[0]} points (center={center}, radius={radius:.3f})")

    dense_dir = out_dir / f"{n_views}_views" / "dense"
    dense_dir.mkdir(parents=True, exist_ok=True)
    _write_ply(dense_dir / "fused.ply", points, colors)
    print(f"[done] FSGS-ready data at {out_dir}")
    print(f"  images: {len(all_view_ids)}, pose-registered: {len(all_view_ids)}, sparse points: {points.shape[0]}")
    return train_ids, test_ids, init_source


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-dir", default="/data/Re-feem/datasets/dtu/scan1")
    parser.add_argument("--n-views", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="experiments/outputs/fsgs_data/dtu_scan1")
    args = parser.parse_args()
    prepare_dtu_for_fsgs(Path(args.scan_dir), args.n_views, args.seed, Path(args.out_dir))  # (train_ids, test_ids, init_source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
