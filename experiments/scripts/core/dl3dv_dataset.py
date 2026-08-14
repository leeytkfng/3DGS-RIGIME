#!/usr/bin/env python3
"""DL3DV(`transforms.json` + `images_8/`) 로더 — dtu_dataset.py/re10k_dataset.py의 DL3DV 버전.

카메라 규약(DepthSplat 공식 변환 스크립트 `src/scripts/convert_dl3dv_train.py::load_metadata`
기준, 2026-08-12 코드 직접 확인 — 우리가 다운받은 pilot 25 scene도 같은 nerfstudio
`transforms.json` 포맷이므로 그대로 재현):

- `w,h,fl_x,fl_y,cx,cy`는 **원본 3840x2160 기준**이고, 우리가 실제로 가진 이미지는
  `images_8/`(8x downsample, ~480x270)이다. intrinsics를 `fl_x/w, fl_y/h, cx/w, cy/h`로
  정규화한 뒤, 실제 이미지 픽셀 크기를 곱해 픽셀 K를 만든다(RE10K와 동일한 패턴).
- `frame["transform_matrix"]`는 **blender(nerfstudio) c2w**다. `opencv_c2w = c2w @
  diag(1,-1,-1,1)`(카메라 로컬 축을 Y-up,-Z-forward -> OpenCV Y-down,Z-forward로 변환)
  후 역행렬을 취하면 COLMAP images.txt가 기대하는 world-to-camera R|t가 된다.
- 우리 pilot 파일에는 top-level `applied_transform`(scene 전체에 적용된 global world-frame
  회전/반사, 3x4)도 있다. 2026-08-12 실측 확인: 이 transform은 **모든 카메라에 똑같이
  적용되는 global rigid transform이라 COLMAP triangulation/overlap 결과(점 개수 등)에는
  전혀 영향이 없다**(적용 여부와 무관하게 동일 SfM point 수 나옴, 상대 기하가 안 바뀌므로
  당연한 결과). 그래도 DepthSplat 원 학습 데이터의 좌표계에 더 가깝게 두기 위해 기본적으로
  undo 하고 진행한다(추후 DL3DV C1-b warm-start를 붙일 때 이 선택이 중요해질 수 있음).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from dtu_dataset import resize_and_crop  # 이름은 dtu_dataset이지만 dataset-agnostic 유틸(image+K만 받음)

_BLENDER_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


def load_metadata(scene_dir: Path) -> dict:
    import json

    return json.loads((scene_dir / "transforms.json").read_text())


def _applied_transform_4x4(meta: dict) -> np.ndarray:
    mat = np.eye(4)
    mat[:3, :4] = np.array(meta["applied_transform"])
    return mat


def load_views(
    scene_dir: Path,
    meta: dict,
    frame_indices: list[int],
    undo_applied_transform: bool = True,
    target_shape: tuple[int, int] | None = None,
) -> list[dict[str, object]]:
    """`dtu_dataset.load_scan()`/`re10k_dataset.load_views()`와 같은 형태의 dict list.

    frame_indices는 `meta["frames"]` 리스트 안의 위치(0-based)다. target_shape=(h_out,w_out)를
    주면 `dtu_dataset.resize_and_crop()`으로 리사이즈한다(FF warm-start 비교용, RE10K와 동일 패턴).
    """

    store_w, store_h = meta["w"], meta["h"]
    saved_fx = meta["fl_x"] / store_w
    saved_fy = meta["fl_y"] / store_h
    saved_cx = meta["cx"] / store_w
    saved_cy = meta["cy"] / store_h

    applied = _applied_transform_4x4(meta) if undo_applied_transform else np.eye(4)
    image_dir = scene_dir / "images_8"

    views = []
    for idx in frame_indices:
        frame = meta["frames"][idx]
        image_name = Path(frame["file_path"]).name
        pil_image = Image.open(image_dir / image_name).convert("RGB")
        width, height = pil_image.size
        image = np.asarray(pil_image, dtype=np.float32) / 255.0

        K = np.array(
            [[saved_fx * width, 0.0, saved_cx * width], [0.0, saved_fy * height, saved_cy * height], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        c2w = applied @ np.array(frame["transform_matrix"]) @ _BLENDER_TO_OPENCV
        w2c = np.linalg.inv(c2w)
        R = w2c[:3, :3]
        t = w2c[:3, 3]
        center = -R.T @ t

        if target_shape is not None:
            image, K = resize_and_crop(image, K, target_shape)
            height, width = image.shape[:2]

        views.append(
            {
                "view_id": idx,
                "image_name": image_name,
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
