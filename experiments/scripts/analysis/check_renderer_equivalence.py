#!/usr/bin/env python3
"""C1-b 렌더 등가성 gate(§5.8) — ps3 env(gsplat 있음)에서 실행.

mvsplat_runner.py(mvsplat env)가 저장한 `gaussians.pt`(FF Gaussian 출력)와
`render_reference.pt`(MVSplat 자체 decoder가 만든 render + 그 render에 쓴 카메라)를 읽어서:
1. FF Gaussians를 gsplat 파라미터화로 변환한다(ff_gaussian_convert).
2. 같은 카메라로 gsplat rasterization을 돌려 재렌더링한다.
3. 두 render를 픽셀 단위로 비교해 config.c1b.renderer_equivalence_tolerance 이내인지 확인한다.

두 conda env(mvsplat=torch 2.1.2, ps3=torch 2.2.2+gsplat)가 호환되지 않아 한 프로세스에서
둘 다 못 돌리므로, mvsplat_runner.py 실행 결과물을 디스크로 넘겨받는 2단계 구조를 쓴다
(체크포인트를 통한 cross-env hand-off는 이 프로젝트의 기존 패턴과 동일).

통과 기준: config의 `c1b.renderer_equivalence_tolerance`(기본 0.0001)를 view별 MSE(픽셀
[0,1] 범위)의 상한으로 해석한다. 통과 못 하면 C1-b(전체)를 진행하지 않는다(§5.8 명시 규칙).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from ff_gaussian_convert import gaussians_to_gsplat_params  # noqa: E402


def render_gsplat(params: dict, viewmat: torch.Tensor, K: torch.Tensor, width: int, height: int, sh_degree: int, device: torch.device) -> torch.Tensor:
    from gsplat import rasterization

    colors = torch.cat([params["sh0"], params["shN"]], dim=1)
    render, _, _ = rasterization(
        means=params["means"],
        quats=params["quats"] / params["quats"].norm(dim=-1, keepdim=True),
        scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]),
        colors=colors,
        viewmats=viewmat[None].to(device),
        Ks=K[None].to(device),
        width=width,
        height=height,
        sh_degree=sh_degree,
        packed=False,
    )
    return render[0].permute(2, 0, 1).clamp(0.0, 1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check FF->gsplat conversion renderer equivalence (ps3 env).")
    parser.add_argument("--checkpoint-dir", required=True, help="mvsplat_runner.py 출력 checkpoints/<scene>/<...>/ 디렉토리")
    parser.add_argument("--tolerance", type=float, default=None, help="MSE 상한(legacy). 지정하면 --tolerance-psnr보다 우선.")
    parser.add_argument(
        "--tolerance-psnr",
        type=float,
        default=33.0,
        help="view별 PSNR 하한(dB). 2026-08-12 DTU 2-view 실측(35.6~42.0dB, 서로 다른 두 CUDA "
        "rasterizer 간 정상 수치오차)에 근거해 §5.8 잠정 기준으로 제안된 값 — 파일럿 전 확정 필요.",
    )
    parser.add_argument("--sh-degree", type=int, default=4, help="MVSplat 기본 config sh_degree(§5.2 실측)")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    gaussians_raw = torch.load(ckpt_dir / "gaussians.pt", map_location="cpu")
    reference = torch.load(ckpt_dir / "render_reference.pt", map_location="cpu")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    means = gaussians_raw["means"][0]  # (N,3), batch=1
    covariances = gaussians_raw["covariances"][0]
    harmonics = gaussians_raw["harmonics"][0]
    opacities = gaussians_raw["opacities"][0]
    params = gaussians_to_gsplat_params(means, covariances, harmonics, opacities, device)

    height, width = reference["image_shape"]
    num_views = reference["pred"].shape[0]
    dtu_scale = reference["dtu_scale_factor"]

    results = []
    for i in range(num_views):
        cam2world = reference["extrinsics"][0, i]
        viewmat = torch.linalg.inv(cam2world).float()
        intrinsics_norm = reference["intrinsics"][0, i]
        K = torch.eye(3, dtype=torch.float32)
        K[0, 0] = intrinsics_norm[0, 0] * width
        K[1, 1] = intrinsics_norm[1, 1] * height
        K[0, 2] = intrinsics_norm[0, 2] * width
        K[1, 2] = intrinsics_norm[1, 2] * height

        gsplat_render = render_gsplat(params, viewmat, K, width, height, args.sh_degree, device)
        mvsplat_render = reference["pred"][i].to(device)

        mse = torch.mean((gsplat_render - mvsplat_render) ** 2).clamp_min(1e-10).item()
        psnr = -10.0 * torch.log10(torch.tensor(mse)).item()
        max_abs_diff = (gsplat_render - mvsplat_render).abs().max().item()
        if args.tolerance is not None:
            passed = mse <= args.tolerance
            criterion = f"mse<={args.tolerance}"
        else:
            passed = psnr >= args.tolerance_psnr
            criterion = f"psnr>={args.tolerance_psnr}dB"
        results.append({
            "view_id": reference["view_ids"][i],
            "mse": mse,
            "psnr": psnr,
            "max_abs_diff": max_abs_diff,
            "passed": passed,
        })
        print(f"[view {reference['view_ids'][i]}] mse={mse:.6f} psnr={psnr:.2f}dB max_abs_diff={max_abs_diff:.4f} "
              f"({criterion}) -> {'PASS' if passed else 'FAIL'}")

    all_passed = all(r["passed"] for r in results)
    summary = {
        "checkpoint_dir": str(ckpt_dir),
        "tolerance_mse": args.tolerance,
        "tolerance_psnr_db": args.tolerance_psnr if args.tolerance is None else None,
        "sh_degree": args.sh_degree,
        "dtu_scale_factor": dtu_scale,
        "results": results,
        "gate_passed": all_passed,
    }
    out_path = ckpt_dir / "renderer_equivalence_gate.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[gate] overall: {'PASS' if all_passed else 'FAIL'}")
    print(f"[done] written to {out_path}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
