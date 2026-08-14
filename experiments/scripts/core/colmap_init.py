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

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap

from dtu_dataset import DTUCamera, load_camera


@dataclass
class CameraParams:
    """데이터셋 무관 카메라 표현 — DTU/RE10K 등 어떤 로더에서 오든 이 형태로 맞추면
    `triangulate_sfm_points_from_cameras()`를 공용으로 쓸 수 있다."""

    K: np.ndarray  # (3,3), pixel-space intrinsics
    R: np.ndarray  # (3,3), world-to-camera rotation
    t: np.ndarray  # (3,), world-to-camera translation
    width: int
    height: int


def _write_known_pose_reconstruction_generic(
    sparse_dir: Path,
    camera_by_name: dict[str, CameraParams],
    id_map: dict[str, tuple[int, int]],  # image_name -> (image_id, camera_id)
) -> None:
    """이름별로 고정된 pose를 COLMAP text 포맷(images.txt/cameras.txt)으로 쓴다.

    DTU처럼 전체 view가 같은 rig(같은 width/height)를 공유한다고 가정하지 않는다 —
    RE10K는 scene마다, 심지어 같은 scene 안에서도 원본 프레임 해상도가 다를 수 있어
    카메라마다 독립적인 width/height/K를 쓴다.
    """

    cameras_lines = ["# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[fx,fy,cx,cy]"]
    images_lines = ["# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME", "# (POINTS2D는 triangulate_points가 채운다)"]

    written_cameras = set()
    for name, (image_id, camera_id) in id_map.items():
        cam = camera_by_name[name]
        if camera_id not in written_cameras:
            fx, fy, cx, cy = cam.K[0, 0], cam.K[1, 1], cam.K[0, 2], cam.K[1, 2]
            cameras_lines.append(f"{camera_id} PINHOLE {cam.width} {cam.height} {fx} {fy} {cx} {cy}")
            written_cameras.add(camera_id)

        qx, qy, qz, qw = pycolmap.Rotation3d(cam.R).quat
        tx, ty, tz = cam.t
        images_lines.append(f"{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {name}")
        images_lines.append("")

    (sparse_dir / "cameras.txt").write_text("\n".join(cameras_lines) + "\n", encoding="utf-8")
    (sparse_dir / "images.txt").write_text("\n".join(images_lines) + "\n", encoding="utf-8")
    (sparse_dir / "points3D.txt").write_text("", encoding="utf-8")


def triangulate_sfm_points_from_cameras(
    image_dir: Path,
    image_names: list[str],
    camera_by_name: dict[str, CameraParams],
    workdir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """known-pose triangulation 공용 코어. 데이터셋별 로더가 image_dir(디스크에 실제
    이미지 파일이 있어야 함)와 camera_by_name만 준비하면 된다.

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
    sparse_in = workdir / "sparse_input"
    sparse_out = workdir / "sparse_triangulated"
    sparse_in.mkdir(parents=True, exist_ok=True)
    sparse_out.mkdir(parents=True, exist_ok=True)

    pycolmap.extract_features(
        str(db_path),
        str(image_dir),
        image_names=image_names,
        camera_mode=pycolmap.CameraMode.PER_IMAGE,
        camera_model="PINHOLE",
    )

    db = pycolmap.Database.open(str(db_path))
    id_map = {}
    for img in db.read_all_images():
        id_map[img.name] = (img.image_id, img.camera_id)
        # DB가 EXIF 기반으로 추측한 intrinsics를 실제 known pose의 K로 덮어써서
        # geometric verification(matching)이 정확한 K를 쓰게 한다.
        cam = camera_by_name[img.name]
        db_camera = db.read_camera(img.camera_id)
        db_camera.params = np.array([cam.K[0, 0], cam.K[1, 1], cam.K[0, 2], cam.K[1, 2]])
        db.update_camera(db_camera)
    db.close()

    _write_known_pose_reconstruction_generic(sparse_in, camera_by_name, id_map)

    pycolmap.match_exhaustive(str(db_path))

    # 극단적으로 sparse한 view 조합(예: 2-view인데 두 view가 거의 안 겹침)에서는
    # SIFT 매칭이 0개가 나올 수 있다. 이 경우 pycolmap.triangulate_points()는
    # 부드러운 예외가 아니라 COLMAP 내부 fatal check("LoadDatabase() failed")로
    # 죽어서 러너 프로세스 전체가 죽는다 (2026-08-10 batch run에서 scan30/103/110이
    # 이렇게 실패했다). 매칭 0개면 triangulation을 시도하지 않고 바로 fallback 신호를
    # 반환해 호출측(MIN_SFM_POINTS 체크)이 random-sphere init으로 넘어가게 한다.
    db = pycolmap.Database.open(str(db_path))
    has_matches = db.num_matches() > 0
    db.close()
    if not has_matches:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)

    reconstruction = pycolmap.Reconstruction()
    reconstruction.read_text(str(sparse_in))
    try:
        result = pycolmap.triangulate_points(reconstruction, str(db_path), str(image_dir), str(sparse_out))
    except (ValueError, RuntimeError) as exc:
        # 위에서 못 잡은 다른 COLMAP 내부 실패에 대한 안전망.
        print(f"[colmap_init] triangulate_points failed, falling back to random init: {exc}")
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)

    if result.num_points3D() == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)

    xyz = np.stack([p.xyz for p in result.points3D.values()])
    rgb = np.stack([p.color for p in result.points3D.values()]).astype(np.uint8)
    return xyz, rgb


def triangulate_sfm_points(
    scan_dir: Path,
    view_ids: list[int],
    workdir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """DTU 전용 진입점(기존 signature 유지). scan_dir/images, scan_dir/cameras에서
    DTU calibration을 읽어 CameraParams로 변환한 뒤 공용 코어를 호출한다."""

    image_path = scan_dir / "images"
    calibration_dir = scan_dir / "cameras"
    image_names = [f"{v:03d}.png" for v in view_ids]
    camera_by_name = {}
    for view_id, name in zip(view_ids, image_names):
        cam: DTUCamera = load_camera(calibration_dir, view_id)
        camera_by_name[name] = CameraParams(K=cam.K, R=cam.R, t=cam.t, width=1600, height=1200)

    return triangulate_sfm_points_from_cameras(image_path, image_names, camera_by_name, workdir)
