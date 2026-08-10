#!/usr/bin/env python3
"""Pose-given track용 COLMAP SfM point triangulation (초기화 전용).

이 파일의 목적:
- vanilla_3dgs_runner.py의 random-in-sphere placeholder init을 실제 COLMAP triangulation으로
  교체한다. 계획서(overall.md §8 STEP1, §5.2)는 optimization의 초기화가 COLMAP SfM이어야 한다고
  명시한다.
- Pose-given track이므로 COLMAP이 pose를 "추정"하지 않는다. 대신 pycolmap.extract_features +
  match_exhaustive로 2D correspondence만 얻고, DTU calibration이 제공하는 고정된 pose로
  `pycolmap.triangulate_points()`(COLMAP CLI의 `point_triangulator`에 해당)를 호출해 3D point만
  얻는다. 카메라 pose 자체는 절대 갱신되지 않는다.

주의:
- 오직 학습에 쓰는 view(입력 view)만 triangulation에 넣는다. held-out test view를 넣으면
  test 정보가 초기화에 새는 leakage가 된다.
- 이 파일은 GT point cloud(stl)를 전혀 참조하지 않는다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pycolmap

from dtu_dataset import DTUCamera, load_camera


def _write_known_pose_reconstruction(
    sparse_dir: Path,
    cameras: dict[int, DTUCamera],
    id_map: dict[str, tuple[int, int]],  # image_name -> (image_id, camera_id)
) -> None:
    """DTU calibration으로 고정된 pose를 COLMAP text 포맷(images.txt/cameras.txt)으로 쓴다."""

    cameras_lines = ["# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[fx,fy,cx,cy]"]
    images_lines = ["# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME", "# (POINTS2D는 triangulate_points가 채운다)"]

    written_cameras = set()
    for name, (image_id, camera_id) in id_map.items():
        view_id = int(name.split(".")[0])
        cam = cameras[view_id]
        if camera_id not in written_cameras:
            fx, fy, cx, cy = cam.K[0, 0], cam.K[1, 1], cam.K[0, 2], cam.K[1, 2]
            cameras_lines.append(f"{camera_id} PINHOLE 1600 1200 {fx} {fy} {cx} {cy}")
            written_cameras.add(camera_id)

        qx, qy, qz, qw = pycolmap.Rotation3d(cam.R).quat
        tx, ty, tz = cam.t
        images_lines.append(f"{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {name}")
        images_lines.append("")

    (sparse_dir / "cameras.txt").write_text("\n".join(cameras_lines) + "\n", encoding="utf-8")
    (sparse_dir / "images.txt").write_text("\n".join(images_lines) + "\n", encoding="utf-8")
    (sparse_dir / "points3D.txt").write_text("", encoding="utf-8")


def triangulate_sfm_points(
    scan_dir: Path,
    view_ids: list[int],
    workdir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """view_ids(학습 view만)로 known-pose triangulation을 수행해 SfM point를 반환한다.

    예상 결과:
    - points_xyz: (N, 3) float64
    - points_rgb: (N, 3) uint8
    - view가 너무 적어 triangulation이 실패/빈약하면 (0,3) 배열을 반환한다
      (호출측에서 random init으로 fallback해야 한다).
    """

    workdir.mkdir(parents=True, exist_ok=True)
    db_path = workdir / "colmap.db"
    if db_path.exists():
        db_path.unlink()
    image_path = scan_dir / "images"
    calibration_dir = scan_dir / "cameras"
    sparse_in = workdir / "sparse_input"
    sparse_out = workdir / "sparse_triangulated"
    sparse_in.mkdir(parents=True, exist_ok=True)
    sparse_out.mkdir(parents=True, exist_ok=True)

    image_names = [f"{v:03d}.png" for v in view_ids]
    cameras = {v: load_camera(calibration_dir, v) for v in view_ids}

    pycolmap.extract_features(
        str(db_path),
        str(image_path),
        image_names=image_names,
        camera_mode=pycolmap.CameraMode.PER_IMAGE,
        camera_model="PINHOLE",
    )

    db = pycolmap.Database.open(str(db_path))
    id_map = {}
    for img in db.read_all_images():
        id_map[img.name] = (img.image_id, img.camera_id)
        # DB가 EXIF 기반으로 추측한 intrinsics를 실제 DTU calibration으로 덮어써서
        # geometric verification(matching)이 정확한 K를 쓰게 한다.
        view_id = int(img.name.split(".")[0])
        cam = cameras[view_id]
        db_camera = db.read_camera(img.camera_id)
        db_camera.params = np.array([cam.K[0, 0], cam.K[1, 1], cam.K[0, 2], cam.K[1, 2]])
        db.update_camera(db_camera)
    db.close()

    _write_known_pose_reconstruction(sparse_in, cameras, id_map)

    pycolmap.match_exhaustive(str(db_path))

    reconstruction = pycolmap.Reconstruction()
    reconstruction.read_text(str(sparse_in))
    result = pycolmap.triangulate_points(reconstruction, str(db_path), str(image_path), str(sparse_out))

    if result.num_points3D() == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)

    xyz = np.stack([p.xyz for p in result.points3D.values()])
    rgb = np.stack([p.color for p in result.points3D.values()]).astype(np.uint8)
    return xyz, rgb
