#!/usr/bin/env python3
"""정성적 비교 그림(DL3DV + ReSplat 확장판)의 GT/DepthSplat/Vanilla3DGS 3장을 만든다.

`render_qualitative_gsplat.py`의 DL3DV 버전. `ps3` env에서 실행. 재실행 없음 — 이미 완료된
C1-a 체크포인트만 다시 렌더링.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))


def save_png(tensor_chw: torch.Tensor, path: Path) -> None:
    arr = (tensor_chw.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def render_gsplat(params: dict, viewmat: torch.Tensor, K: torch.Tensor, width: int, height: int, sh_degree: int) -> torch.Tensor:
    from gsplat import rasterization

    with torch.no_grad():
        colors = torch.cat([params["sh0"], params["shN"]], dim=1)
        render, _, _ = rasterization(
            means=params["means"],
            quats=params["quats"] / params["quats"].norm(dim=-1, keepdim=True),
            scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]),
            colors=colors,
            viewmats=viewmat[None],
            Ks=K[None],
            width=width,
            height=height,
            sh_degree=sh_degree,
            packed=False,
        )
    return render[0].permute(2, 0, 1).clamp(0.0, 1.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, help="DL3DV scene key(폴더명 hash)")
    parser.add_argument("--view-count", type=int, required=True)
    parser.add_argument("--budget", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--c1a-root", default="experiments/outputs/dl3dv_c1a_main")
    parser.add_argument("--out-dir", default="experiments/outputs/qualitative_comparison_dl3dv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    c1a_root = Path(args.c1a_root)
    tag = f"{args.scene}_{args.view_count}view_{int(args.budget)}s"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 공용 target camera: DepthSplat render_reference.pt ---
    ref = torch.load(c1a_root / "checkpoints" / args.scene / f"{args.view_count}view" / "render_reference.pt", map_location="cpu")
    i = args.target_index
    view_id = ref["view_ids"][i]
    c2w = ref["extrinsics"][0, i]
    K_norm = ref["intrinsics"][0, i]
    h, w = ref["image_shape"]
    w2c = torch.inverse(c2w)
    K_pixel = K_norm.clone()
    K_pixel[0, :] *= w
    K_pixel[1, :] *= h

    # --- GT ---
    from dl3dv_dataset import load_metadata, load_views

    scene_dir = Path("/data/Re-feem/datasets/dl3dv") / args.scene
    meta = load_metadata(scene_dir)
    gt_view = load_views(scene_dir, meta, [view_id], target_shape=(h, w))[0]
    gt_tensor = torch.from_numpy(gt_view["image"]).permute(2, 0, 1).float()
    save_png(gt_tensor, out_dir / f"{tag}_GT.png")

    # --- DepthSplat: 이미 렌더된 pred 그대로 ---
    save_png(ref["pred"][i], out_dir / f"{tag}_DepthSplat.png")

    # --- Vanilla3DGS: budget 시점 checkpoint 다시 렌더링 ---
    traj_path = c1a_root / "vanilla_runs" / args.scene / "logs" / f"dl3dv_{args.scene}_c1a_Vanilla3DGS_{args.view_count}view_seed{args.seed}.json"
    trajectory = json.loads(traj_path.read_text())
    row = next(r for r in trajectory if r["wall_clock"] == args.budget)
    ckpt = torch.load(row["checkpoint_path"], map_location="cpu")
    params = {k: v.to(device) for k, v in ckpt.items()}
    d_sh = params["shN"].shape[1] + 1
    sh_degree = round(math.sqrt(d_sh) - 1)
    vanilla_render = render_gsplat(params, w2c.to(device), K_pixel.to(device), w, h, sh_degree)
    save_png(vanilla_render, out_dir / f"{tag}_Vanilla3DGS.png")

    # --- FSGS/ReSplat 렌더링용 camera 정보 저장 ---
    camera_json = {
        "scene": args.scene, "view_count": args.view_count, "budget": args.budget, "seed": args.seed,
        "view_id": view_id, "width": w, "height": h,
        "w2c": w2c.numpy().tolist(), "K_norm": K_norm.numpy().tolist(), "c2w": c2w.numpy().tolist(),
    }
    (out_dir / f"{tag}_camera.json").write_text(json.dumps(camera_json, indent=2))

    print(f"[done] {tag}: GT/DepthSplat/Vanilla3DGS PNG + camera.json written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
