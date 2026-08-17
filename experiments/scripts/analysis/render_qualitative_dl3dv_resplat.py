#!/usr/bin/env python3
"""정성적 비교 그림(DL3DV)의 ReSplat 렌더 — `resplat` conda env에서 실행.

`resplat_dl3dv_runner.py`와 같은 context view(v2 overlap summary)를 쓰되, target camera는
`render_qualitative_dl3dv_gsplat.py`가 저장한 camera.json(=DepthSplat과 동일한 target)을
그대로 써서 네 방법(DepthSplat/Vanilla3DGS/FSGS/ReSplat) 전부 같은 시점을 렌더링하게 만든다.
ReSplat 추론 자체는 결정론적 단일 forward pass라 이것도 사실상 재실행 비용이 거의 없다
(scene당 ~1.2초, `run_resplat_dl3dv.py`가 이미 계산한 것과 별개로 target camera만 다르게
한 번 더 도는 것).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "experiments/scripts/runners"))


def save_png(tensor_chw: torch.Tensor, path: Path) -> None:
    arr = (tensor_chw.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-json", required=True)
    parser.add_argument("--overlap-summary", default="experiments/outputs/dl3dv_overlap_v2/all_scenes_summary.json")
    parser.add_argument("--num-refine", type=int, default=4)
    parser.add_argument("--checkpoint", default="/data/Re-feem/code/resplat/pretrained/resplat-base-dl3dv-256x448-view8-1934a04c.pth")
    parser.add_argument("--resplat-repo", default="/data/Re-feem/code/resplat")
    parser.add_argument("--out-dir", default="experiments/outputs/qualitative_comparison_dl3dv")
    args = parser.parse_args()

    import resplat_dl3dv_runner as R

    resplat_repo = Path(args.resplat_repo)
    R._add_repo_paths(resplat_repo)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cam_info = json.loads(Path(args.camera_json).read_text())
    scene, view_count = cam_info["scene"], cam_info["view_count"]
    width, height = cam_info["width"], cam_info["height"]
    target_c2w = torch.tensor(cam_info["c2w"], dtype=torch.float32)
    target_K = torch.tensor(cam_info["K_norm"], dtype=torch.float32)

    overlap_data = json.loads(Path(args.overlap_summary).read_text())
    row = next(r for r in overlap_data if r["scene"] == scene and r["view_count"] == view_count)
    context_ids = row["context_indices"]

    from dl3dv_dataset import load_metadata

    scene_dir = Path("/data/Re-feem/datasets/dl3dv") / scene
    meta = load_metadata(scene_dir)
    context_images, context_c2w, context_K = R.load_frames(scene_dir, meta, context_ids)

    Vc = len(context_ids)
    all_c2w = torch.cat([context_c2w, target_c2w[None]], dim=0)
    mid_idx = Vc // 2
    all_c2w = R.camera_normalization(context_c2w[mid_idx : mid_idx + 1], all_c2w)
    context_c2w_aligned = all_c2w[:Vc]
    target_c2w_aligned = all_c2w[Vc:]

    batch = {
        "context": {
            "image": context_images.unsqueeze(0).to(device),
            "extrinsics": context_c2w_aligned.unsqueeze(0).to(device),
            "intrinsics": context_K.unsqueeze(0).to(device),
            "near": torch.full((1, Vc), R.NEAR, device=device),
            "far": torch.full((1, Vc), R.FAR, device=device),
            "index": torch.arange(Vc, device=device).unsqueeze(0),
        },
        "target": {
            "extrinsics": target_c2w_aligned.unsqueeze(0).to(device),
            "intrinsics": target_K.unsqueeze(0).unsqueeze(0).to(device),
            "near": torch.full((1, 1), R.NEAR, device=device),
            "far": torch.full((1, 1), R.FAR, device=device),
        },
        "scene": [scene],
    }

    encoder, decoder = R.build_model(resplat_repo, "dl3dv", args.checkpoint, args.num_refine, R.IMAGE_SHAPE, device)

    with torch.no_grad():
        gaussians_out = encoder(batch["context"], global_step=0, deterministic=False)
        condition_features = gaussians_out.get("condition_features") if isinstance(gaussians_out, dict) else None
        gaussians = gaussians_out["gaussians"] if isinstance(gaussians_out, dict) else gaussians_out
        if args.num_refine > 0 and condition_features is not None:
            refine_output = encoder.forward_update(
                batch["context"], batch["target"], condition_features, gaussians, decoder, None,
            )
            gaussians = refine_output["gaussian"][-1]
        output = decoder.forward(
            gaussians, batch["target"]["extrinsics"], batch["target"]["intrinsics"],
            batch["target"]["near"], batch["target"]["far"], R.IMAGE_SHAPE,
        )

    pred = output.color[0, 0].clamp(0, 1)
    tag = f"{scene}_{view_count}view_{int(cam_info['budget'])}s"
    out_dir = Path(args.out_dir)
    save_png(pred, out_dir / f"{tag}_ReSplat.png")
    print(f"[done] {tag}_ReSplat.png written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
