#!/usr/bin/env python3
"""정성적 비교 그림(§Results 정성적 비교 todo)의 GT/MVSplat/Vanilla3DGS 3장을 만든다.

`ps3` conda env(gsplat 있음)에서 실행. 이미 완료된 C1-a 체크포인트만 다시 렌더링한다 —
새 학습 없음, 재실행 아님.

- **GT**: RE10K 데이터셋에서 직접 로드.
- **MVSplat**: `checkpoints/{scene}/{view_count}view/render_reference.pt`에 이미 렌더된
  `pred`를 그대로 crop해서 쓴다(재렌더링 불필요 — feed-forward는 결정론적 단일 추론이라
  저장된 값 자체가 최종 결과).
- **Vanilla3DGS**: `vanilla_runs/{scene}/logs/*_seed{seed}.json` 트래젝토리에서 지정
  budget의 `checkpoint_path`를 찾아 gsplat으로 다시 렌더링(`vanilla_3dgs_runner.py`의
  rasterization 호출과 동일 파라미터화).

동일 target camera(= render_reference.pt의 extrinsics/intrinsics)로 세 장 다 렌더링해
공정한 비교가 되게 한다. FSGS는 이 스크립트가 아니라 `render_qualitative_fsgs.py`(fsgs
conda env)가 같은 camera 정보를 받아 별도로 렌더링한다 — 이 스크립트가 camera json도
같이 저장해준다.
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
    parser.add_argument("--scene", required=True, help="RE10K scene key, e.g. 0588138dfec165a1")
    parser.add_argument("--view-count", type=int, required=True)
    parser.add_argument("--budget", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-index", type=int, default=0, help="render_reference.pt의 target 중 몇 번째")
    parser.add_argument("--c1a-root", default="experiments/outputs/re10k_c1a_main")
    parser.add_argument("--re10k-subset-index", default="experiments/outputs/re10k_main_subset/re10k_main_subset.json")
    parser.add_argument("--out-dir", default="experiments/outputs/qualitative_comparison")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    c1a_root = Path(args.c1a_root)
    tag = f"{args.scene}_{args.view_count}view_{int(args.budget)}s"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 공용 target camera: MVSplat render_reference.pt ---
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
    from re10k_dataset import get_scene_item, load_views

    subset = json.loads(Path(args.re10k_subset_index).read_text())
    entry = subset[args.scene]
    item = get_scene_item(Path("/data/Re-feem/datasets/re10k/test") / entry["chunk_file"], args.scene)
    gt_view = load_views(item, [view_id], target_shape=(h, w))[0]
    gt_tensor = torch.from_numpy(gt_view["image"]).permute(2, 0, 1).float()
    save_png(gt_tensor, out_dir / f"{tag}_GT.png")

    # --- MVSplat: 이미 렌더된 pred 그대로 ---
    mvsplat_pred = ref["pred"][i]
    save_png(mvsplat_pred, out_dir / f"{tag}_MVSplat.png")

    # --- Vanilla3DGS: budget 시점 checkpoint 다시 렌더링 ---
    traj_path = c1a_root / "vanilla_runs" / args.scene / "logs" / f"re10k_{args.scene}_c1a_Vanilla3DGS_{args.view_count}view_seed{args.seed}.json"
    trajectory = json.loads(traj_path.read_text())
    row = next(r for r in trajectory if r["wall_clock"] == args.budget)
    ckpt = torch.load(row["checkpoint_path"], map_location="cpu")
    params = {k: v.to(device) for k, v in ckpt.items()}
    d_sh = params["shN"].shape[1] + 1
    sh_degree = round(math.sqrt(d_sh) - 1)
    vanilla_render = render_gsplat(params, w2c.to(device), K_pixel.to(device), w, h, sh_degree)
    save_png(vanilla_render, out_dir / f"{tag}_Vanilla3DGS.png")

    # --- FSGS 렌더링용 camera 정보 저장(별도 env에서 이어서 처리) ---
    camera_json = {
        "scene": args.scene, "view_count": args.view_count, "budget": args.budget, "seed": args.seed,
        "view_id": view_id, "width": w, "height": h,
        "w2c": w2c.numpy().tolist(), "K_norm": K_norm.numpy().tolist(),
    }
    (out_dir / f"{tag}_camera.json").write_text(json.dumps(camera_json, indent=2))

    print(f"[done] {tag}: GT/MVSplat/Vanilla3DGS PNG + camera.json written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
