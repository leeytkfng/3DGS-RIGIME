#!/usr/bin/env python3
"""Depth Anything V2 Metric wrapper — C2 depth back-projection의 depth 소스.

전용 conda env `depth`(torch/torchvision/transformers/pillow만 설치, gsplat 등과 분리)에서만
import한다. overall.md §5.9 2026-08-15 항목 참고 — VGGT 대신 이 모델을 쓰기로 한 이유:
우리 트랙은 pose-given이라 VGGT의 pose 추정이 불필요하고, 이 모델은 known intrinsics로 바로
back-projection 가능한 metric depth를 직접 출력해 파이프라인이 더 단순하다.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"


class DepthEstimator:
    def __init__(self, device: str = "cuda", model_id: str = MODEL_ID):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.torch = torch
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()

    def predict(self, image_rgb01: np.ndarray) -> np.ndarray:
        """image_rgb01: (H, W, 3) float [0,1]. 반환: (H, W) metric depth(m), 입력과 같은 해상도로
        bilinear upsample된 값(모델 자체 출력 해상도는 입력보다 작은 경우가 많다)."""
        image_uint8 = (np.clip(image_rgb01, 0.0, 1.0) * 255.0).astype(np.uint8)
        pil_image = Image.fromarray(image_uint8)
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        depth = outputs.predicted_depth  # (1, H', W')
        h, w = image_rgb01.shape[:2]
        depth_full = self.torch.nn.functional.interpolate(
            depth.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False
        )
        return depth_full.squeeze().cpu().numpy().astype(np.float32)
