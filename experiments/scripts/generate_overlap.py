#!/usr/bin/env python3
"""Generate zero-included overlap reports from SfM visibility."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

try:
    from protocol_utils import build_overlap_report
except ModuleNotFoundError:  # Allows package imports from tests.
    from experiments.scripts.protocol_utils import build_overlap_report


def _read_view_list(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    # The subset file is intentionally plain text so sampling scripts can write
    # exactly the selected input views without depending on a dataset-specific schema.
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_visibility_json(path: Path) -> dict[str, set[str]]:
    """Load {view_id: [point_id, ...]} visibility JSON."""

    # This compact format is useful when visibility comes from a custom SfM
    # preprocessing step rather than directly from COLMAP text exports.
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("visibility JSON must be an object mapping view ids to point id lists")

    return {str(view_id): {str(point_id) for point_id in point_ids} for view_id, point_ids in payload.items()}


def load_colmap_images_txt(path: Path, view_name_key: str = "name") -> dict[str, set[str]]:
    """Parse COLMAP images.txt and return visible POINT3D_ID sets per image."""

    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    data_lines = [line for line in lines if line and not line.startswith("#")]
    view_points: dict[str, set[str]] = {}

    if len(data_lines) % 2 != 0:
        raise ValueError("COLMAP images.txt should contain image metadata and point lines in pairs")

    for index in range(0, len(data_lines), 2):
        # COLMAP stores each image as two lines: metadata, then flattened
        # keypoint observations in (x, y, POINT3D_ID) triplets.
        image_fields = data_lines[index].split()
        point_fields = data_lines[index + 1].split()
        if len(image_fields) < 10:
            raise ValueError(f"Malformed COLMAP image line: {data_lines[index]}")

        image_id = image_fields[0]
        image_name = image_fields[9]
        view_id = image_name if view_name_key == "name" else image_id

        if len(point_fields) % 3 != 0:
            raise ValueError(f"Malformed COLMAP point line for image {view_id}")

        visible = set()
        for offset in range(2, len(point_fields), 3):
            point3d_id = point_fields[offset]
            # COLMAP uses -1 for unmatched 2D observations. Those are not SfM
            # tracks and must not contribute to co-visibility.
            if point3d_id != "-1":
                visible.add(point3d_id)
        view_points[view_id] = visible

    return view_points


def subset_views(view_points: dict[str, set[str]], view_ids: Iterable[str] | None) -> dict[str, set[str]]:
    if view_ids is None:
        return view_points

    # Failing fast here prevents a silent mismatch between the sampled sparse-view
    # condition and the overlap report used as the regime-map x-axis.

    missing = [view_id for view_id in view_ids if view_id not in view_points]
    if missing:
        raise KeyError(f"Requested views not found in visibility input: {missing}")

    return {view_id: view_points[view_id] for view_id in view_ids}


def write_pairwise_csv(pairwise: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["view_i", "view_j", "overlap", "shared_points", "points_i", "points_j", "connected"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in pairwise:
            writer.writerow({key: row[key] for key in fieldnames})


def generate_overlap_report(
    view_points: dict[str, set[str]],
    output_dir: Path,
    min_common_points: int = 1,
    scene: str | None = None,
    view_count_label: str | None = None,
) -> dict[str, object]:
    # build_overlap_report applies the zero-included non-edge policy. Keeping
    # metadata beside the numbers makes later threshold-freezing auditable.
    report = build_overlap_report(view_points, min_common_points=min_common_points)
    report["metadata"] = {
        "scene": scene,
        "view_count_label": view_count_label,
        "min_common_points": min_common_points,
        "non_edge_policy": "zero_included",
        "primary_statistic": "mean_overlap",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_pairwise_csv(report["pairwise"], output_dir / "pairwise_overlap.csv")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an overlap report from SfM visibility.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--visibility-json", type=Path, help="JSON mapping view ids to visible SfM point ids.")
    source.add_argument("--colmap-images", type=Path, help="COLMAP text export images.txt path.")
    parser.add_argument("--views", type=Path, help="Optional newline-delimited view subset file.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for summary.json and pairwise_overlap.csv.")
    parser.add_argument("--min-common-points", type=int, default=1, help="Shared point threshold below which a pair is recorded as zero.")
    parser.add_argument("--scene", help="Scene id stored in summary metadata.")
    parser.add_argument("--view-count-label", help="View-count/overlap sampling label stored in metadata.")
    parser.add_argument("--colmap-view-key", choices=["name", "id"], default="name", help="Use COLMAP image name or image id as view id.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.visibility_json:
        view_points = load_visibility_json(args.visibility_json)
    else:
        view_points = load_colmap_images_txt(args.colmap_images, view_name_key=args.colmap_view_key)

    view_points = subset_views(view_points, _read_view_list(args.views))
    report = generate_overlap_report(
        view_points=view_points,
        output_dir=args.output_dir,
        min_common_points=args.min_common_points,
        scene=args.scene,
        view_count_label=args.view_count_label,
    )

    summary = report["summary"]
    print(f"Wrote overlap report to: {args.output_dir}")
    print(f"Views: {summary['view_count']} | pairs: {summary['pair_count']} | zero-pair ratio: {summary['zero_pair_ratio']:.3f}")
    print(f"Mean overlap: {summary['mean_overlap']:.6f} | q25: {summary['q25_overlap']:.6f}")
    if summary["has_isolated_view"]:
        print("Isolated-view flag: true; separate this sample from the main aggregate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
