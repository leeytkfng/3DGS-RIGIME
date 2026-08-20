#!/usr/bin/env python3
"""정성적 비교 그림의 FSGS 렌더 — `fsgs` conda env에서 실행.

`render_qualitative_gsplat.py`가 저장한 `{tag}_camera.json`(w2c, 정규화 K, width/height)을
읽어 FSGS 자체 `PseudoCamera`/`render()`로 같은 target camera를 렌더링한다. 이미 저장된
FSGS point_cloud.ply를 다시 읽을 뿐 학습은 하지 않는다 — 재실행 아님.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image


def save_png(tensor_chw: torch.Tensor, path: Path) -> None:
    arr = (tensor_chw.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-json", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--c1a-root", default="experiments/outputs/re10k_c1a_main")
    parser.add_argument("--fsgs-repo", default="/data/Re-feem/code/fsgs")
    parser.add_argument("--out-dir", default="experiments/outputs/qualitative_comparison")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(args.fsgs_repo).resolve()))
    from utils.graphics_utils import focal2fov
    from scene.cameras import PseudoCamera
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    from arguments import PipelineParams
    import argparse as _argparse

    cam_info = json.loads(Path(args.camera_json).read_text())
    scene, view_count, budget, seed = cam_info["scene"], cam_info["view_count"], cam_info["budget"], args.seed
    width, height = cam_info["width"], cam_info["height"]
    w2c = np.array(cam_info["w2c"])
    K_norm = np.array(cam_info["K_norm"])

    R = w2c[:3, :3].T
    T = w2c[:3, 3]
    fx_pixel, fy_pixel = K_norm[0, 0] * width, K_norm[1, 1] * height
    FoVx = focal2fov(fx_pixel, width)
    FoVy = focal2fov(fy_pixel, height)

    device = torch.device("cuda")
    cam = PseudoCamera(R=R, T=T, FoVx=FoVx, FoVy=FoVy, width=width, height=height)

    c1a_root = Path(args.c1a_root)
    traj_path = c1a_root / "fsgs_runs" / scene / "logs" / f"re10k_{scene}_c1a_FSGS_{view_count}view_seed{seed}.json"
    trajectory = json.loads(traj_path.read_text())
    row = next(r for r in trajectory if r["wall_clock"] == budget)

    parser_fsgs = _argparse.ArgumentParser()
    pp = PipelineParams(parser_fsgs)
    pipe = pp.extract(parser_fsgs.parse_args([]))

    gaussians = GaussianModel(SimpleNamespace(sh_degree=3))  # FSGS 기본값(fsgs_runner.py와 동일)
    gaussians.load_ply(row["checkpoint_path"])
    background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)

    with torch.no_grad():
        image = render(cam, gaussians, pipe, background)["render"]

    tag = f"{scene}_{view_count}view_{int(budget)}s"
    out_dir = Path(args.out_dir)
    save_png(image, out_dir / f"{tag}_FSGS.png")
    print(f"[done] {tag}_FSGS.png written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
