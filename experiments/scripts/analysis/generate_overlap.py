#!/usr/bin/env python3
"""SfM visibility에서 zero-included overlap report를 생성하는 CLI.

이 파일의 목적:
- COLMAP `images.txt` 또는 간단한 visibility JSON을 읽는다.
- view pair별 co-visibility와 전체 summary를 만든다.
- 예상 출력은 `summary.json`과 `pairwise_overlap.csv`이다.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable

try:
    # 스크립트를 `python experiments/scripts/analysis/generate_overlap.py`처럼 직접 실행할 때 사용된다.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
    from protocol_utils import build_overlap_report
except ModuleNotFoundError:  # tests에서 package import 형태로 실행할 때 사용된다.
    from experiments.scripts.core.protocol_utils import build_overlap_report


def _read_view_list(path: Path | None) -> list[str] | None:
    """선택할 view 목록 파일을 읽는다.

    목적:
    - sparse-view sampling 결과를 한 줄에 view 하나씩 저장해두고 그대로 재사용한다.

    예상 결과:
    - path가 None이면 전체 view를 쓰기 위해 None을 반환한다.
    - 파일이 있으면 빈 줄을 제외한 view id list를 반환한다.
    """

    if path is None:
        return None
    # 일부 데이터셋은 별도 schema가 다르므로, 가장 단순한 newline text를 표준으로 둔다.
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_visibility_json(path: Path) -> dict[str, set[str]]:
    """`{view_id: [point_id, ...]}` 형태의 visibility JSON을 읽는다.

    목적:
    - COLMAP이 아닌 전처리기에서 visibility를 뽑았을 때도 같은 overlap 계산기를 쓰기 위함.

    예상 결과:
    - 모든 view id와 point id를 문자열로 통일한 dict를 반환한다.
    - 문자열 통일 덕분에 int/string id가 섞여도 pairwise intersection이 안정적이다.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("visibility JSON must be an object mapping view ids to point id lists")

    return {str(view_id): {str(point_id) for point_id in point_ids} for view_id, point_ids in payload.items()}


def load_colmap_images_txt(path: Path, view_name_key: str = "name") -> dict[str, set[str]]:
    """COLMAP text export의 `images.txt`에서 view별 visible POINT3D_ID를 뽑는다.

    목적:
    - SfM point track을 이용해 co-visibility overlap을 계산한다.

    입력:
    - view_name_key="name": 이미지 파일명을 view id로 사용한다.
    - view_name_key="id": COLMAP image id를 view id로 사용한다.

    예상 결과:
    - `{view_id: {POINT3D_ID, ...}}` 형태의 dict를 반환한다.
    - COLMAP의 unmatched observation인 POINT3D_ID=-1은 제외한다.
    """

    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    data_lines = [line for line in lines if line and not line.startswith("#")]
    view_points: dict[str, set[str]] = {}

    if len(data_lines) % 2 != 0:
        raise ValueError("COLMAP images.txt should contain image metadata and point lines in pairs")

    for index in range(0, len(data_lines), 2):
        # COLMAP images.txt는 이미지 하나가 두 줄이다.
        # 1번째 줄: IMAGE_ID, quaternion, translation, camera id, image name.
        # 2번째 줄: keypoint 관측치가 (x, y, POINT3D_ID) triplet으로 반복된다.
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
            # -1은 2D feature가 어떤 3D point에도 연결되지 않았다는 뜻이다.
            # co-visibility는 3D point track 공유량이므로 -1은 반드시 제외한다.
            if point3d_id != "-1":
                visible.add(point3d_id)
        view_points[view_id] = visible

    return view_points


def subset_views(view_points: dict[str, set[str]], view_ids: Iterable[str] | None) -> dict[str, set[str]]:
    """전체 visibility에서 sparse-view subset만 골라낸다.

    목적:
    - 2/4/8/12 view sampling 결과와 overlap 계산 입력을 정확히 일치시킨다.

    예상 결과:
    - view_ids가 None이면 전체 dict를 반환한다.
    - 요청한 view가 없으면 즉시 KeyError를 내서 잘못된 실험 조건을 막는다.
    """

    if view_ids is None:
        return view_points

    # 조용히 누락 view를 빼버리면 view count 자체가 바뀐다. 그래서 실패를 빠르게 드러낸다.
    missing = [view_id for view_id in view_ids if view_id not in view_points]
    if missing:
        raise KeyError(f"Requested views not found in visibility input: {missing}")

    return {view_id: view_points[view_id] for view_id in view_ids}


def write_pairwise_csv(pairwise: list[dict[str, object]], path: Path) -> None:
    """pairwise overlap table을 CSV로 저장한다.

    목적:
    - histogram/debug/논문 표 생성을 쉽게 하기 위해 pair별 결과를 평평한 표로 남긴다.

    예상 결과:
    - `pairwise_overlap.csv`가 생성되고 각 row는 하나의 view pair를 의미한다.
    """

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
    """overlap summary와 pairwise table을 파일로 저장한다.

    목적:
    - 파일럿 단계에서 scene/view-count/seed별 overlap report를 누적한다.
    - 나중에 low/high threshold를 고정할 때 metadata와 수치를 함께 추적한다.

    예상 결과:
    - output_dir/summary.json
    - output_dir/pairwise_overlap.csv
    - 반환값은 저장된 report dict와 같다.
    """

    # build_overlap_report 내부에서 non-edge를 0으로 포함하는 핵심 규칙이 적용된다.
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
    """CLI 인자를 정의한다.

    목적:
    - COLMAP 입력과 JSON 입력 중 하나만 받도록 강제한다.
    - output 위치와 view subset을 명시적으로 남기게 한다.

    예상 결과:
    - argparse가 검증한 args를 main에서 사용할 수 있다.
    """

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
    """CLI 진입점.

    실행 흐름:
    1. 입력 visibility를 읽는다.
    2. 선택 view list가 있으면 subset을 적용한다.
    3. summary.json과 pairwise_overlap.csv를 저장한다.
    4. 터미널에는 빠른 sanity check용 요약값을 출력한다.

    예상 결과:
    - 정상 종료 시 0을 반환한다.
    """

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
