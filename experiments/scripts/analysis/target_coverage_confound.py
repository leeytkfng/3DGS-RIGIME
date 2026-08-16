#!/usr/bin/env python3
"""A-2 진단: 우리 co-visibility selector(`core/view_selector.py`)의 high/low overlap 후보가
target-coverage(=target이 context로 얼마나 잘 "둘러싸이는가")를 교란하는지 확인한다.

배경: `select_high_overlap_indices`는 좁은 window 안에서, `select_low_overlap_indices`는 전체
범위에서 같은 FPS를 돌린다. 그러면 low 후보는 구조적으로 더 넓게 퍼지므로, overlap과 무관하게
"target을 convex hull로 감싸거나(bracketing) target 방향에 더 가까운 context가 있을 확률"도
같이 올라갈 수 있다 — 이러면 high/low 비교 결과가 overlap 축이 아니라 이 배치 효과를 재는 것일
수 있다.

**진단 전용 스크립트다. `core/view_selector.py`는 건드리지 않는다** — 결과를 보고 selector를
바꾸는 것은 프로토콜 변경이라 공동 승인이 필요하다(A-2 작업 지시 원문 그대로).

입력: `re10k_overlap_candidates.json`/`dl3dv_overlap_candidates.json`(이미 생성돼 있음, 이 세션
에서 새로 만들지 않는다) + pose만 읽는 경량 로더(이미지 디코딩 없음, `generate_*_overlap_candidates.py`
가 쓰던 것과 동일 패턴).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay, QhullError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from protocol_utils import scene_cluster_bootstrap_ci  # noqa: E402

RE10K_ROOT = Path("/data/Re-feem/datasets/re10k/test")
DL3DV_ROOT = Path("/data/Re-feem/datasets/dl3dv")
BLENDER_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])

_CHUNK_CACHE: dict[str, list[dict]] = {}


def _load_chunk(path: Path) -> list[dict]:
    import torch

    key = str(path)
    if key not in _CHUNK_CACHE:
        _CHUNK_CACHE[key] = torch.load(path, weights_only=False)
    return _CHUNK_CACHE[key]


def _re10k_item(chunk_path: Path, scene_key: str) -> dict:
    for item in _load_chunk(chunk_path):
        if item["key"] == scene_key:
            return item
    raise KeyError(scene_key)


def re10k_center_forward(item: dict, idx: int) -> tuple[np.ndarray, np.ndarray]:
    """이미지 디코딩 없이 pose만 읽어 (camera center, forward direction)을 반환한다."""

    pose = item["cameras"][idx].numpy()
    w2c = pose[6:18].reshape(3, 4)
    R, t = w2c[:, :3].astype(np.float64), w2c[:, 3].astype(np.float64)
    center = -R.T @ t
    forward = R.T @ np.array([0.0, 0.0, 1.0])
    return center, forward / np.linalg.norm(forward)


def dl3dv_center_forward(meta: dict, idx: int) -> tuple[np.ndarray, np.ndarray]:
    applied = np.eye(4)
    if "applied_transform" in meta:
        applied[:3, :4] = np.array(meta["applied_transform"])
    frame = meta["frames"][idx]
    c2w = applied @ np.array(frame["transform_matrix"]) @ BLENDER_TO_OPENCV
    w2c = np.linalg.inv(c2w)
    R, t = w2c[:3, :3], w2c[:3, 3]
    center = -R.T @ t
    forward = R.T @ np.array([0.0, 0.0, 1.0])
    return center, forward / np.linalg.norm(forward)


def is_bracketed(target_center: np.ndarray, context_centers: np.ndarray) -> bool | None:
    """target이 context center들의 convex hull 안에 있는가. 점이 너무 적거나(<=2) 축퇴돼서
    (거의 공면/공선) 판정 불가능하면 None(해당 없음)을 반환한다.

    3D hull이 축퇴되면(Qhull 에러) 주성분 2D 평면에 투영 후 재시도한다(A-2 지시문의 제안 그대로).
    """

    if context_centers.shape[0] < 3:
        return None

    def _contains(points: np.ndarray, query: np.ndarray) -> bool | None:
        try:
            tri = Delaunay(points)
        except QhullError:
            return None
        return bool(tri.find_simplex(query) >= 0)

    result = _contains(context_centers, target_center)
    if result is not None:
        return result

    centroid = context_centers.mean(axis=0)
    centered = np.vstack([context_centers, target_center[None, :]]) - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:2]
    proj_context = (context_centers - centroid) @ basis.T
    proj_target = (target_center - centroid) @ basis.T
    return _contains(proj_context, proj_target)


def min_angular_distance_deg(target_forward: np.ndarray, context_forwards: np.ndarray) -> float:
    dots = np.clip(context_forwards @ target_forward, -1.0, 1.0)
    return float(np.degrees(np.min(np.arccos(dots))))


def condition_metrics(
    target_centers: list[np.ndarray],
    target_forwards: list[np.ndarray],
    context_centers: np.ndarray,
    context_forwards: np.ndarray,
) -> dict[str, float]:
    """한 (scene, view_count, level) 조건에 대해 target view들 평균 지표를 낸다."""

    bracket_flags, angles, centroid_dists, nearest_dists = [], [], [], []
    context_centroid = context_centers.mean(axis=0)
    for tc, tf in zip(target_centers, target_forwards):
        b = is_bracketed(tc, context_centers)
        if b is not None:
            bracket_flags.append(float(b))
        angles.append(min_angular_distance_deg(tf, context_forwards))
        centroid_dists.append(float(np.linalg.norm(tc - context_centroid)))
        nearest_dists.append(float(np.min(np.linalg.norm(context_centers - tc, axis=1))))

    return {
        "bracket_rate": float(np.mean(bracket_flags)) if bracket_flags else float("nan"),
        "bracket_n": len(bracket_flags),
        "min_angle_deg": float(np.mean(angles)),
        "centroid_dist": float(np.mean(centroid_dists)),
        "nearest_dist": float(np.mean(nearest_dists)),
    }


def collect_re10k(candidates_path: Path, subset_path: Path) -> list[dict]:
    candidates = json.loads(candidates_path.read_text())
    subset = json.loads(subset_path.read_text())
    rows = []
    for scene_key, by_view in candidates.items():
        chunk_path = RE10K_ROOT / subset[scene_key]["chunk_file"]
        item = _re10k_item(chunk_path, scene_key)
        for view_count, row in by_view.items():
            target_indices = row["target"]
            t_centers, t_forwards = zip(*(re10k_center_forward(item, i) for i in target_indices))
            for level in ("high", "low"):
                ctx_indices = row[level]["context"]
                c_centers, c_forwards = zip(*(re10k_center_forward(item, i) for i in ctx_indices))
                metrics = condition_metrics(
                    list(t_centers), list(t_forwards), np.stack(c_centers), np.stack(c_forwards)
                )
                rows.append(
                    {
                        "dataset": "RE10K",
                        "scene": scene_key,
                        "view_count": int(view_count),
                        "level": level,
                        **metrics,
                    }
                )
    return rows


def collect_dl3dv(candidates_path: Path) -> list[dict]:
    candidates = json.loads(candidates_path.read_text())
    meta_cache: dict[str, dict] = {}
    rows = []
    for scene, by_view in candidates.items():
        if scene not in meta_cache:
            meta_cache[scene] = json.loads((DL3DV_ROOT / scene / "transforms.json").read_text())
        meta = meta_cache[scene]
        for view_count, row in by_view.items():
            for level in ("high", "low"):
                cell = row[level]
                target_indices = cell["target"]
                ctx_indices = cell["context"]
                t_centers, t_forwards = zip(*(dl3dv_center_forward(meta, i) for i in target_indices))
                c_centers, c_forwards = zip(*(dl3dv_center_forward(meta, i) for i in ctx_indices))
                metrics = condition_metrics(
                    list(t_centers), list(t_forwards), np.stack(c_centers), np.stack(c_forwards)
                )
                rows.append(
                    {
                        "dataset": "DL3DV",
                        "scene": scene,
                        "view_count": int(view_count),
                        "level": level,
                        **metrics,
                    }
                )
    return rows


def summarize(rows: list[dict], dataset: str) -> None:
    view_counts = sorted({r["view_count"] for r in rows if r["dataset"] == dataset})
    print(f"\n=== {dataset}: high vs low, view_count별 ===")
    for vc in view_counts:
        vc_rows = [r for r in rows if r["dataset"] == dataset and r["view_count"] == vc]
        for metric in ("bracket_rate", "min_angle_deg", "centroid_dist", "nearest_dist"):
            high_rows = [r for r in vc_rows if r["level"] == "high" and not np.isnan(r[metric])]
            low_rows = [r for r in vc_rows if r["level"] == "low" and not np.isnan(r[metric])]
            if not high_rows or not low_rows:
                print(f"  [{vc}view] {metric}: 데이터 부족(high n={len(high_rows)}, low n={len(low_rows)})")
                continue
            hi = scene_cluster_bootstrap_ci(high_rows, lambda r: r[metric])
            lo = scene_cluster_bootstrap_ci(low_rows, lambda r: r[metric])
            by_scene_high = {r["scene"]: r[metric] for r in high_rows}
            by_scene_low = {r["scene"]: r[metric] for r in low_rows}
            common = sorted(set(by_scene_high) & set(by_scene_low))
            delta_rows = [{"scene": s, "delta": by_scene_high[s] - by_scene_low[s]} for s in common]
            delta = scene_cluster_bootstrap_ci(delta_rows, lambda r: r["delta"]) if delta_rows else None
            print(
                f"  [{vc}view] {metric}: high={hi['mean']:.3f} [{hi['ci_low']:.3f},{hi['ci_high']:.3f}] "
                f"(n={hi['scene_count']}) vs low={lo['mean']:.3f} [{lo['ci_low']:.3f},{lo['ci_high']:.3f}] "
                f"(n={lo['scene_count']})"
                + (
                    f"  | paired delta(high-low)={delta['mean']:+.3f} [{delta['ci_low']:+.3f},{delta['ci_high']:+.3f}]"
                    if delta
                    else ""
                )
            )


def main() -> int:
    re10k_rows = collect_re10k(
        Path("experiments/outputs/re10k_overlap_candidates/re10k_overlap_candidates.json"),
        Path("experiments/outputs/re10k_main_subset/re10k_main_subset.json"),
    )
    dl3dv_rows = collect_dl3dv(Path("experiments/outputs/dl3dv_overlap_lowhigh/dl3dv_overlap_candidates.json"))

    out_path = Path("experiments/outputs/target_coverage_confound/rows.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(re10k_rows + dl3dv_rows, indent=2), encoding="utf-8")
    print(f"[done] {out_path} ({len(re10k_rows) + len(dl3dv_rows)} rows)")

    summarize(re10k_rows, "RE10K")
    summarize(dl3dv_rows, "DL3DV")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
