#!/usr/bin/env python3
"""DTU smoke용 2/4/8/12 view set과 co-visibility overlap report 생성기.

목적:
- runner들이 쓰는 seed 기반 train-view 선택 규칙을 그대로 재사용해, 실제 smoke run의 입력 view와
  overlap 계산 입력이 어긋나지 않게 한다.
- 각 view count마다 COLMAP known-pose triangulation을 수행하고, triangulated SfM point track에서
  zero-included overlap report를 만든다.

예상 결과:
- output_dir/<scene>/<N>view_seed<S>/views.txt
- output_dir/<scene>/<N>view_seed<S>/visibility.json
- output_dir/<scene>/<N>view_seed<S>/summary.json
- output_dir/<scene>/<N>view_seed<S>/pairwise_overlap.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pycolmap

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from colmap_init import triangulate_sfm_points  # noqa: E402
from protocol_utils import build_overlap_report  # noqa: E402

ALL_DTU_VIEW_IDS = list(range(1, 50))
DTU_TEST_IDS = ALL_DTU_VIEW_IDS[::7]
DTU_TRAIN_POOL = [v for v in ALL_DTU_VIEW_IDS if v not in DTU_TEST_IDS]


def select_runner_views(view_count: int, seed: int) -> list[int]:
    """Vanilla/MVSplat DTU runner와 동일한 seed 기반 view 선택 규칙."""

    rng = np.random.default_rng(seed)
    return sorted(rng.choice(DTU_TRAIN_POOL, size=min(view_count, len(DTU_TRAIN_POOL)), replace=False).tolist())


def visibility_from_reconstruction(model_dir: Path) -> dict[str, set[str]]:
    """COLMAP triangulated binary model에서 image별 visible POINT3D_ID를 추출한다."""

    reconstruction = pycolmap.Reconstruction(str(model_dir))
    visibility: dict[str, set[str]] = {}
    for image in reconstruction.images.values():
        visible = set()
        for point2d in image.points2D:
            point_id = int(point2d.point3D_id)
            if point_id != -1:
                visible.add(str(point_id))
        visibility[image.name] = visible
    return visibility


def write_pairwise_csv(pairwise: list[dict[str, object]], path: Path) -> None:
    import csv

    fieldnames = ["view_i", "view_j", "overlap", "shared_points", "points_i", "points_j", "connected"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in pairwise:
            writer.writerow({key: row[key] for key in fieldnames})


def write_report(visibility: dict[str, set[str]], output_dir: Path, scene: str, label: str, min_common_points: int) -> dict[str, object]:
    report = build_overlap_report(visibility, min_common_points=min_common_points)
    report["metadata"] = {
        "scene": scene,
        "view_count_label": label,
        "min_common_points": min_common_points,
        "non_edge_policy": "zero_included",
        "primary_statistic": "mean_overlap",
        "source": "pycolmap sparse_triangulated image point tracks",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    serializable_visibility = {k: sorted(v) for k, v in visibility.items()}
    (output_dir / "visibility.json").write_text(json.dumps(serializable_visibility, indent=2), encoding="utf-8")
    write_pairwise_csv(report["pairwise"], output_dir / "pairwise_overlap.csv")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate DTU smoke view sets and overlap reports.")
    parser.add_argument("--scan-dir", default="/data/Re-feem/datasets/dtu/scan1")
    parser.add_argument("--scene", default="dtu_scan1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--view-counts", type=int, nargs="+", default=[2, 4, 8, 12])
    parser.add_argument("--output-dir", default="experiments/outputs_smoke_20260811/overlap")
    parser.add_argument("--colmap-work-dir", default="experiments/outputs_smoke_20260811/colmap_work")
    parser.add_argument("--min-common-points", type=int, default=1)
    args = parser.parse_args()

    scan_dir = Path(args.scan_dir)
    output_root = Path(args.output_dir)
    colmap_root = Path(args.colmap_work_dir)

    summaries = []
    for view_count in args.view_counts:
        label = f"{view_count}view_seed{args.seed}"
        view_ids = select_runner_views(view_count, args.seed)
        report_dir = output_root / args.scene / label
        workdir = colmap_root / args.scene / label
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "views.txt").write_text("\n".join(f"{v:03d}.png" for v in view_ids) + "\n", encoding="utf-8")
        (report_dir / "view_ids.json").write_text(json.dumps(view_ids, indent=2), encoding="utf-8")

        # triangulate_sfm_points는 view가 너무 안 겹치면 빈 point를 반환한다. 이 경우에도
        # 각 view를 빈 visibility로 넣어 zero-overlap report를 남긴다.
        points, _colors = triangulate_sfm_points(scan_dir, view_ids, workdir)
        model_dir = workdir / "sparse_triangulated"
        if points.shape[0] > 0 and (model_dir / "images.bin").exists():
            visibility = visibility_from_reconstruction(model_dir)
        else:
            visibility = {f"{v:03d}.png": set() for v in view_ids}

        report = write_report(visibility, report_dir, args.scene, label, args.min_common_points)
        summary = report["summary"]
        row = {
            "scene": args.scene,
            "label": label,
            "view_ids": view_ids,
            "sfm_points": int(points.shape[0]),
            "mean_overlap": summary["mean_overlap"],
            "q25_overlap": summary["q25_overlap"],
            "zero_pair_ratio": summary["zero_pair_ratio"],
            "has_isolated_view": summary["has_isolated_view"],
            "report_dir": str(report_dir),
        }
        summaries.append(row)
        print(
            f"[{label}] views={view_ids} sfm_points={points.shape[0]} "
            f"mean_overlap={summary['mean_overlap']:.4f} zero_pair_ratio={summary['zero_pair_ratio']:.3f}"
        )

    summary_path = output_root / args.scene / f"seed{args.seed}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"[done] summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
