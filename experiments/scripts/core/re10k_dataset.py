#!/usr/bin/env python3
"""RE10K(.torch chunk) 로더 — dtu_dataset.py의 RE10K 버전.

이 파일의 목적:
- pixelSplat/MVSplat 계열 `.torch` chunk 포맷(list[{key, url, timestamps, cameras[N,18], images}])에서
  scene 하나를 찾고, 지정한 frame index들만 디스크에 이미지로 풀어서
  colmap_init.triangulate_sfm_points_from_cameras()가 바로 쓸 수 있는 CameraParams를 만든다.

카메라 규약(MVSplat `src/dataset/dataset_re10k.py::convert_poses` 기준, 2026-08-12 실측 확인):
- poses[:4] = fx, fy, cx, cy — 이미지 너비/높이로 **정규화된** 값(0~1 범위). 실제 픽셀 K로
  쓰려면 fx*width, cx*width, fy*height, cy*height로 스케일해야 한다.
- poses[4:6]은 쓰지 않는다.
- poses[6:18] = world-to-camera 3x4 행렬을 flatten한 것. COLMAP images.txt가 기대하는
  convention과 동일(W2C)이므로 별도 inverse 없이 그대로 R/t로 쓴다.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dtu_dataset import resize_and_crop  # 이름은 dtu_dataset이지만 dataset-agnostic 유틸(image+K만 받음)

_CHUNK_CACHE: dict[str, list[dict]] = {}


def load_chunk(chunk_path: Path) -> list[dict]:
    key = str(chunk_path)
    if key not in _CHUNK_CACHE:
        _CHUNK_CACHE[key] = torch.load(chunk_path, weights_only=False)
    return _CHUNK_CACHE[key]


def get_scene_item(chunk_path: Path, scene_key: str) -> dict:
    for item in load_chunk(chunk_path):
        if item["key"] == scene_key:
            return item
    raise KeyError(f"scene {scene_key} not found in {chunk_path}")


def extract_frames_to_disk(
    item: dict,
    frame_indices: list[int],
    out_dir: Path,
) -> dict[str, "CameraParams"]:
    """지정한 frame들을 out_dir/<frame_idx:03d>.png로 저장하고 CameraParams를 만든다.

    pycolmap(colmap_init.CameraParams)이 필요한 함수라 mvsplat/depthsplat conda env에는
    없을 수 있다 — 그래서 import를 함수 안으로 늦춰서, overlap 계산에 안 쓰이는
    `load_views()`/`get_scene_item()`만 쓰는 호출부(예: mvsplat_re10k_runner.py)가
    pycolmap 없이도 이 모듈을 import할 수 있게 한다.
    """

    from colmap_init import CameraParams

    out_dir.mkdir(parents=True, exist_ok=True)
    camera_by_name: dict[str, CameraParams] = {}
    for frame_idx in frame_indices:
        name = f"{frame_idx:03d}.png"
        raw = item["images"][frame_idx]
        image = Image.open(BytesIO(raw.numpy().tobytes())).convert("RGB")
        image.save(out_dir / name)
        width, height = image.size

        pose = item["cameras"][frame_idx].numpy()
        fx, fy, cx, cy = pose[:4]
        K = np.array(
            [
                [fx * width, 0.0, cx * width],
                [0.0, fy * height, cy * height],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        w2c = pose[6:18].reshape(3, 4)
        R = w2c[:, :3].astype(np.float64)
        t = w2c[:, 3].astype(np.float64)

        camera_by_name[name] = CameraParams(K=K, R=R, t=t, width=width, height=height)

    return camera_by_name


def load_views(
    item: dict,
    frame_indices: list[int],
    target_shape: tuple[int, int] | None = None,
) -> list[dict[str, object]]:
    """`dtu_dataset.load_scan()`과 같은 형태의 dict list를 만든다 — `vanilla_3dgs_runner.py`의
    `build_camera_tensors()`/training loop가 데이터셋 종류와 무관하게 그대로 돌게 하기 위함.

    DTU와 달리 RE10K 카메라는 이미 world-to-camera 3x4를 그대로 제공하고(§변환 규약은 모듈
    docstring 참고) scale factor가 필요 없다 — `mvsplat_re10k_probe.py`가 이미 이렇게
    검증했다(DTU_SCALE_FACTOR 같은 보정이 RE10K 경로엔 없음).
    """

    views = []
    for frame_idx in frame_indices:
        raw = item["images"][frame_idx]
        pil_image = Image.open(BytesIO(raw.numpy().tobytes())).convert("RGB")
        width, height = pil_image.size
        image = np.asarray(pil_image, dtype=np.float32) / 255.0

        pose = item["cameras"][frame_idx].numpy()
        fx, fy, cx, cy = pose[:4]
        K = np.array(
            [[fx * width, 0.0, cx * width], [0.0, fy * height, cy * height], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        w2c = pose[6:18].reshape(3, 4)
        R = w2c[:, :3].astype(np.float64)
        t = w2c[:, 3].astype(np.float64)
        center = -R.T @ t

        if target_shape is not None:
            image, K = resize_and_crop(image, K, target_shape)
            height, width = image.shape[:2]

        views.append(
            {
                "view_id": frame_idx,
                "image": image,
                "K": K,
                "R": R,
                "t": t,
                "center": center,
                "width": width,
                "height": height,
            }
        )
    return views
