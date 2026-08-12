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

from colmap_init import CameraParams

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
) -> dict[str, CameraParams]:
    """지정한 frame들을 out_dir/<frame_idx:03d>.png로 저장하고 CameraParams를 만든다."""

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
