#!/usr/bin/env python3
"""DTU (roboimagedata.compute.dtu.dk SampleSet) scan 로더.

이 파일의 목적:
- DTU calibration 폴더의 3x4 projection matrix(pos_XXX.txt)를 K, world-to-camera R|t로 분해한다.
- Pose-given track이므로 COLMAP을 거치지 않고 DTU가 제공하는 calibration을 그대로 신뢰한다.
- vanilla_3dgs_runner.py가 요구하는 형태(dict of camera + image tensor)로 scan을 읽어온다.

주의:
- Gaussian 초기화는 이 파일이 만들지 않는다. 여기서는 카메라 기하만 다룬다.
  초기화용 point sampling은 vanilla_3dgs_runner.py에서 카메라 기하로부터 추정한
  scene bounding sphere를 사용하며, GT point cloud(stl)는 절대 참조하지 않는다.
  (실험 프로토콜상 optimization 초기화는 COLMAP SfM을 써야 하고, 지금은 COLMAP triangulation을
  아직 연결하지 않았기 때문에 random-in-sphere init을 임시로 쓴다. §5.2/§5.4 참고.)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class DTUCamera:
    """하나의 DTU view에 대한 pinhole camera 정보.

    좌표계 정의:
    - world_to_cam: X_cam = R @ X_world + t
    - K: pixel intrinsics (3x3)
    """

    view_id: int
    K: np.ndarray  # (3, 3)
    R: np.ndarray  # (3, 3), world-to-camera rotation
    t: np.ndarray  # (3,), world-to-camera translation
    center: np.ndarray  # (3,), camera center in world coords


def _rq3(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """3x3 행렬을 RQ 분해한다 (M = R @ Q, R은 상삼각, Q는 직교).

    numpy에는 RQ가 없어서 QR을 뒤집는 표준 트릭을 쓴다.
    """

    flipped = np.flipud(matrix).T
    q, r = np.linalg.qr(flipped)
    r = np.flipud(r.T)
    q = q.T[::-1, :]
    return r[:, ::-1], q


def decompose_projection(projection: np.ndarray) -> DTUCamera:
    """3x4 projection matrix P = K[R|t]를 K, R, t, camera center로 분해한다.

    예상 결과:
    - K의 대각선(fx, fy)이 양수가 되도록 부호를 고정한다 (RQ 분해는 부호가 임의).
    - det(R) = 1이 되도록 보정한다.
    """

    m = projection[:, :3]
    p4 = projection[:, 3]

    k, r = _rq3(m)

    # RQ 분해는 K의 대각 부호가 임의로 나올 수 있다. fx, fy를 양수로 고정하고
    # 부호를 R로 흡수시켜서 K@R = M이 계속 성립하도록 만든다.
    sign_fix = np.diag(np.sign(np.diag(k)))
    k = k @ sign_fix
    r = sign_fix @ r

    if np.linalg.det(r) < 0:
        r = -r
        k = -k
        # k의 대각을 다시 양수로 맞춘다 (k 전체 부호를 뒤집었으므로 대각도 뒤집힘).
        sign_fix2 = np.diag(np.sign(np.diag(k)))
        k = k @ sign_fix2
        r = sign_fix2 @ r

    k = k / k[2, 2]
    t = np.linalg.solve(k, p4)
    center = -r.T @ t

    return DTUCamera(view_id=-1, K=k, R=r, t=t, center=center)


def load_projection_matrix(path: Path) -> np.ndarray:
    """DTU calibration 폴더의 pos_XXX.txt(3x4 행렬 3줄)를 읽는다."""

    values = [float(x) for x in path.read_text(encoding="utf-8").split()]
    if len(values) != 12:
        raise ValueError(f"Expected 12 values (3x4 matrix) in {path}, got {len(values)}")
    return np.array(values, dtype=np.float64).reshape(3, 4)


def load_camera(calibration_dir: Path, view_id: int) -> DTUCamera:
    """view_id(1-based)의 camera를 calibration 폴더에서 읽어 분해한다."""

    projection = load_projection_matrix(calibration_dir / f"pos_{view_id:03d}.txt")
    camera = decompose_projection(projection)
    return DTUCamera(view_id=view_id, K=camera.K, R=camera.R, t=camera.t, center=camera.center)


def load_image(images_dir: Path, view_id: int) -> np.ndarray:
    """view_id의 RGB 이미지를 [0, 1] float32 HxWx3 array로 읽는다."""

    image = Image.open(images_dir / f"{view_id:03d}.png").convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def estimate_scene_sphere(cameras: Sequence[DTUCamera]) -> tuple[np.ndarray, float]:
    """카메라들의 시선(principal axis)이 가장 가깝게 모이는 지점을 scene 중심으로 추정한다.

    목적:
    - GT point cloud 없이도 Gaussian 초기화를 놓을 대략적인 위치/반경을 얻는다.
    - 여러 직선(각 카메라의 시선)에 가장 가까운 한 점을 최소자승으로 구하는
      표준적인 "closest point to multiple lines" 공식을 사용한다.

    예상 결과:
    - center: world 좌표계에서 추정된 scene 중심.
    - radius: 각 카메라 중심에서 center까지 거리의 중앙값 기반 추정 반경.
    """

    a_sum = np.zeros((3, 3))
    b_sum = np.zeros(3)
    for camera in cameras:
        # camera forward(+z)가 world에서 가리키는 방향.
        direction = camera.R.T @ np.array([0.0, 0.0, 1.0])
        direction = direction / np.linalg.norm(direction)
        projector = np.eye(3) - np.outer(direction, direction)
        a_sum += projector
        b_sum += projector @ camera.center

    center = np.linalg.solve(a_sum, b_sum)
    distances = [float(np.linalg.norm(camera.center - center)) for camera in cameras]
    radius = float(np.median(distances)) * 0.35
    return center, radius


def load_scan(
    scan_dir: Path,
    view_ids: Sequence[int],
) -> list[dict[str, object]]:
    """scan_dir(예: .../dtu/scan1) 아래 images/, cameras/ 에서 지정한 view들을 읽는다.

    예상 결과:
    - 각 원소는 {"view_id", "image", "K", "R", "t", "center", "width", "height"}를 갖는 dict.
    """

    images_dir = scan_dir / "images"
    calibration_dir = scan_dir / "cameras"

    views = []
    for view_id in view_ids:
        camera = load_camera(calibration_dir, view_id)
        image = load_image(images_dir, view_id)
        height, width = image.shape[:2]
        views.append(
            {
                "view_id": view_id,
                "image": image,
                "K": camera.K,
                "R": camera.R,
                "t": camera.t,
                "center": camera.center,
                "width": width,
                "height": height,
            }
        )
    return views
