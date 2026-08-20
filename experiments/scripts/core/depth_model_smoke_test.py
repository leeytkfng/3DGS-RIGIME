#!/usr/bin/env python3
"""Depth Anything V2 Metric 설치 확인 — DTU 이미지 1장에 대해 실제로 depth map을 뽑아본다.
C2 백프로젝션 초기화 경로에 쓰기 전 모델이 정상 동작하는지만 확인하는 용도.

실행: /opt/conda/envs/depth/bin/python3 core/depth_model_smoke_test.py
(전용 conda env `depth` 필요 — torch/torchvision/transformers/pillow만 설치된 가벼운 env,
ps3 등 기존 env와 분리해 버전 충돌을 피함. overall.md §5.9 2026-08-15 항목 참고.)
"""
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
IMG_PATH = "/data/Re-feem/datasets/dtu/scan1/images/001.png"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[setup] device={device}")

t0 = time.time()
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).to(device).eval()
print(f"[setup] model loaded in {time.time()-t0:.1f}s")

image = Image.open(IMG_PATH).convert("RGB")
print(f"[data] image size={image.size}")

inputs = processor(images=image, return_tensors="pt").to(device)
t1 = time.time()
with torch.no_grad():
    outputs = model(**inputs)
print(f"[infer] wall_clock={time.time()-t1:.3f}s")

depth = outputs.predicted_depth  # (1, H', W') in meters (metric variant)
depth_np = depth.squeeze().cpu().numpy()
print(f"[result] depth shape={depth_np.shape}, dtype={depth_np.dtype}")
print(f"[result] depth min={depth_np.min():.3f} max={depth_np.max():.3f} mean={depth_np.mean():.3f} (meters, model's own scale)")
print(f"[result] any NaN: {np.isnan(depth_np).any()}, any inf: {np.isinf(depth_np).any()}")
print("[done] depth model smoke test passed" if not np.isnan(depth_np).any() else "[FAIL] NaNs in output")
