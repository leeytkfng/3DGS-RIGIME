#!/usr/bin/env python3
"""C4-a 파일럿 — 기성 신호(feed-forward가 이미 출력하는 opacity)가 refinement 중 Gaussian이
얼마나 움직이는지를 예측하는가? (재실행 0, 기존 C1-b 체크포인트만 사용)

배경: Diff3R(arXiv 2604.01030)는 per-Gaussian uncertainty를 *학습*해 방향별 LM damping을 얻고,
비겹침 영역일수록 그 값이 커짐을 보인다(Fig. 5/6). 우리 C4-a 질문은 "그 학습이 반드시
필요한가 — 이미 학습 없이 나오는 신호로도 같은 걸 볼 수 있는가"이다. MVSplat/DepthSplat/FSGS는
명시적 confidence 출력이 없다는 걸 이미 확인했으므로(§III), 여기서는 MVSplat이 어차피 내는
**opacity**를 대체 신호 후보로 시험한다 — opacity는 렌더링 손실만으로 학습되지만, 불확실한
지점(깊이 경계, 반투명, 저겹침 등)일수록 낮게 나올 개연성이 있다.

측정 대상: refinement 전(iter0, feed-forward 원본) 대비 refinement 후(60s) Gaussian 위치
변화. **핵심 기술적 난점**: densification이 켜져 있어(C1-b 기본 설정) iter0→60s 사이 Gaussian
개수가 바뀌고(예: 131072→189163) 인덱스가 더 이상 대응하지 않는다 — clone/split/prune를
거치므로 "몇 번 Gaussian이 몇 번이 됐는지"를 직접 알 방법이 로그에 없다. 이 파일럿은 그
근사로 **최근접 이웃(nearest-neighbor) 매칭**을 쓴다: iter0의 각 Gaussian을 60s 시점 점군에서
3D로 가장 가까운 점에 매칭하고 그 거리를 "이동량"으로 삼는다. 이는 진짜 계보 추적이 아니라
근사이며, 특히 이동량이 큰 영역일수록 매칭 오차도 커질 수 있다는 한계가 있다 — 깨끗한 답을
얻으려면 densification=off C1-b 실행(인덱스가 안 바뀜, `vanilla_3dgs_runner.py`에 이미 있는
`--densification off` 플래그로 가능)이 필요하다. 이 스크립트는 그 실행 전 방향성 신호가
있는지만 저비용으로 확인하는 용도다.

입력: `experiments/outputs/re10k_c1b_scaleup*/vanilla_runs/<scene>/checkpoints/.../{ff_warm_start_baseline_iter0,budget_60.0s_iter*}.pt`
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

C1B_DIRS = {
    "RE10K_2view": "experiments/outputs/re10k_c1b_scaleup",
    "RE10K_4view": "experiments/outputs/re10k_c1b_scaleup_4view",
}


def load_pair(scene_dir: Path) -> tuple[dict, dict] | None:
    ckpt_glob = list(scene_dir.glob("checkpoints/*/*/ff_warm_start_baseline_iter0.pt"))
    if not ckpt_glob:
        return None
    p0_path = ckpt_glob[0]
    cond_dir = p0_path.parent
    p60_candidates = list(cond_dir.glob("budget_60.0s_iter*.pt"))
    if not p60_candidates:
        return None
    p0 = torch.load(p0_path, map_location="cpu", weights_only=False)
    p60 = torch.load(p60_candidates[0], map_location="cpu", weights_only=False)
    return p0, p60


def analyze_scene(p0: dict, p60: dict) -> dict:
    mu0 = p0["means"].numpy()
    mu60 = p60["means"].numpy()
    raw_op0 = p0["opacities"].numpy()
    # opacities는 러너에 따라 pre-sigmoid(logit) 또는 post-sigmoid로 저장될 수 있어 범위로 판별.
    op0 = 1 / (1 + np.exp(-raw_op0)) if (raw_op0.min() < 0 or raw_op0.max() > 1) else raw_op0

    tree = cKDTree(mu60)
    dist, _ = tree.query(mu0, k=1)

    corr = float(np.corrcoef(op0, dist)[0, 1])
    order = np.argsort(op0)
    n = len(op0)
    q = max(1, n // 4)
    low_op_dist = float(dist[order[:q]].mean())
    high_op_dist = float(dist[order[-q:]].mean())

    return {
        "n_iter0": int(len(mu0)),
        "n_60s": int(len(mu60)),
        "opacity_range": [float(op0.min()), float(op0.max())],
        "corr_opacity_vs_nn_dist": corr,
        "low_opacity_quartile_mean_dist": low_op_dist,
        "high_opacity_quartile_mean_dist": high_op_dist,
        "ratio_low_over_high": low_op_dist / high_op_dist if high_op_dist > 0 else None,
    }


def main() -> int:
    results = []
    for label, root in C1B_DIRS.items():
        root_path = Path(root) / "vanilla_runs"
        if not root_path.exists():
            continue
        for scene_dir in sorted(root_path.glob("*")):
            pair = load_pair(scene_dir)
            if pair is None:
                continue
            p0, p60 = pair
            row = {"condition": label, "scene": scene_dir.name}
            row.update(analyze_scene(p0, p60))
            results.append(row)
            print(
                f"[{label}] {scene_dir.name}: n0={row['n_iter0']} n60={row['n_60s']} "
                f"corr={row['corr_opacity_vs_nn_dist']:+.3f} "
                f"low_q_dist={row['low_opacity_quartile_mean_dist']:.4f} "
                f"high_q_dist={row['high_opacity_quartile_mean_dist']:.4f} "
                f"ratio={row['ratio_low_over_high']:.2f}x"
            )

    if results:
        corrs = [r["corr_opacity_vs_nn_dist"] for r in results]
        ratios = [r["ratio_low_over_high"] for r in results if r["ratio_low_over_high"]]
        print(f"\n[summary] n_scenes={len(results)}  mean_corr={np.mean(corrs):+.3f}  mean_ratio(low/high)={np.mean(ratios):.2f}x")

    out_path = Path("experiments/outputs/c4a_pilot/opacity_vs_movement.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[done] {out_path} ({len(results)} scenes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
