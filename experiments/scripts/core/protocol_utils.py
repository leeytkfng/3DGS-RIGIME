#!/usr/bin/env python3
"""Sparse-view 3DGS 실험 프로토콜에서 공통으로 쓰는 계산 유틸리티.

이 파일의 목적:
- 논문에서 미리 고정한 overlap, checkpoint, 통계 규칙을 코드 한 곳에 모은다.
- 실제 모델 실행 코드가 붙어도 중요한 실험 규칙이 흩어지지 않게 한다.
- 예상 결과는 숫자 report, manifest, 분석 스크립트가 모두 같은 기준을 쓰는 것이다.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations
from statistics import mean
from typing import Callable, Iterable, Mapping, Sequence


# 승패 판정에서 seed 변동성이 너무 작게 나오더라도 최소한 이 정도 PSNR 차이는
# 실용적으로 의미 있는 차이로 본다. 파일럿 이후 config 값으로 조정 가능하다.
PRACTICAL_MIN_PSNR_DELTA = 0.5


@dataclass(frozen=True)
class PairwiseOverlap:
    """두 view 사이의 SfM co-visibility 결과 한 줄.

    목적:
    - pairwise_overlap.csv에 그대로 저장할 수 있는 구조화된 레코드를 만든다.

    예상 결과:
    - connected=True이면 공통 SfM point가 충분한 pair다.
    - connected=False이면 매칭 실패 또는 공통 point 부족 pair이며 overlap은 0이다.
    """

    view_i: str
    view_j: str
    overlap: float
    shared_points: int
    points_i: int
    points_j: int
    connected: bool

    def to_dict(self) -> dict[str, float | int | str | bool]:
        """CSV/JSON 저장을 위해 dataclass를 일반 dict로 변환한다."""

        return asdict(self)


def compute_pairwise_overlaps(
    view_points: Mapping[str, Iterable[str | int]],
    min_common_points: int = 1,
) -> list[PairwiseOverlap]:
    """모든 view pair의 overlap을 계산한다.

    입력:
    - view_points: view id -> 해당 view가 관측한 SfM point id 목록
    - min_common_points: 이 값보다 공통 point가 적으면 연결 실패 pair로 본다.

    중요한 규칙:
    - 매칭 실패 pair를 제외하지 않고 overlap=0으로 포함한다.
    - 이것이 regime map의 x축을 왜곡하지 않기 위한 핵심 규칙이다.

    예상 결과:
    - view가 N개이면 N*(N-1)/2개의 PairwiseOverlap이 반환된다.
    """

    # Iterable이 한 번만 순회 가능한 객체일 수 있으므로 set으로 고정한다.
    point_sets = {view: set(points) for view, points in view_points.items()}
    overlaps: list[PairwiseOverlap] = []

    # sorted를 사용해 출력 순서를 항상 같게 만든다. 그래야 seed/debug 비교가 쉽다.
    for view_i, view_j in combinations(sorted(point_sets), 2):
        points_i = point_sets[view_i]
        points_j = point_sets[view_j]
        shared = len(points_i & points_j)
        denominator = len(points_i) + len(points_j)

        # 논문에서 가장 중요한 overlap 규칙이다.
        # 실패한 매칭을 버리면 low-texture/low-overlap 장면의 overlap이 과대평가된다.
        if shared < min_common_points or denominator == 0:
            overlap = 0.0
            connected = False
        else:
            # Dice 형태의 co-visibility: 2|Pi cap Pj| / (|Pi| + |Pj|)
            overlap = 2.0 * shared / denominator
            connected = True

        overlaps.append(
            PairwiseOverlap(
                view_i=view_i,
                view_j=view_j,
                overlap=overlap,
                shared_points=shared,
                points_i=len(points_i),
                points_j=len(points_j),
                connected=connected,
            )
        )

    return overlaps


def percentile(values: Sequence[float], q: float) -> float:
    """선형 보간 percentile 계산.

    목적:
    - numpy/pandas 없이 q25, median 같은 파일럿 지표를 계산한다.

    예상 결과:
    - values가 비어 있으면 math.nan을 반환한다.
    - q=0.25이면 하위 25% 분위수, q=0.5이면 median이다.
    """

    if not values:
        return math.nan
    if q < 0.0 or q > 1.0:
        raise ValueError("q must be in [0, 1]")

    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]

    rank = q * (len(sorted_values) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[lower]

    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def graph_has_isolated_view(view_ids: Sequence[str], pairwise: Sequence[PairwiseOverlap]) -> bool:
    """입력 view 집합에 고립 view가 있는지 확인한다.

    목적:
    - 고립 view가 있는 sample은 low-overlap보다 더 극단적인 실패 범주일 수 있다.
    - 문서 설계에 따라 메인 집계에서 분리 보고하기 위한 flag다.

    예상 결과:
    - 하나라도 connected pair가 없는 view가 있으면 True를 반환한다.
    """

    degree = {view_id: 0 for view_id in view_ids}
    for item in pairwise:
        if item.connected:
            degree[item.view_i] += 1
            degree[item.view_j] += 1
    return any(value == 0 for value in degree.values())


def aggregate_overlap(
    view_points: Mapping[str, Iterable[str | int]],
    min_common_points: int = 1,
) -> dict[str, float | int | bool]:
    """pairwise overlap을 논문용 summary 지표로 집계한다.

    목적:
    - 파일럿 histogram과 low/high overlap threshold 동결에 필요한 요약값을 만든다.

    예상 결과:
    - mean_overlap: 주 지표. 0 pair를 포함한 전체 pair 평균.
    - q25_overlap: low-overlap 꼬리를 보는 보조 지표.
    - median_overlap_reference: 참고용. 주 지표로 쓰지 않는다.
    - zero_pair_ratio: 실패 pair 비율.
    - has_isolated_view: 메인 집계 분리 여부 flag.
    """

    pairwise = compute_pairwise_overlaps(view_points, min_common_points=min_common_points)
    values = [item.overlap for item in pairwise]
    zero_pairs = sum(1 for item in pairwise if item.overlap == 0.0)

    # 전체 pair 평균이 regime map의 x축이다. median은 sparse 조건에서 0 pair를
    # 가려버릴 수 있으므로 참고 지표로만 남긴다.
    return {
        "view_count": len(view_points),
        "pair_count": len(pairwise),
        "zero_pair_count": zero_pairs,
        "zero_pair_ratio": zero_pairs / len(pairwise) if pairwise else math.nan,
        "mean_overlap": mean(values) if values else math.nan,
        "q25_overlap": percentile(values, 0.25),
        "median_overlap_reference": percentile(values, 0.5),
        "has_isolated_view": graph_has_isolated_view(sorted(view_points), pairwise),
    }


def build_overlap_report(
    view_points: Mapping[str, Iterable[str | int]],
    min_common_points: int = 1,
) -> dict[str, object]:
    """summary와 pairwise table을 함께 담은 overlap report를 만든다.

    목적:
    - `generate_overlap.py`가 JSON/CSV를 저장할 때 같은 계산 결과를 재사용한다.

    예상 결과:
    - 반환 dict는 `summary`와 `pairwise` 키를 갖는다.
    """

    pairwise = compute_pairwise_overlaps(view_points, min_common_points=min_common_points)
    return {
        "summary": aggregate_overlap(view_points, min_common_points=min_common_points),
        "pairwise": [item.to_dict() for item in pairwise],
    }


def classify_delta(delta_psnr: float, tau: float) -> str:
    """FF와 OPT의 PSNR 차이를 win/tie/loss label로 바꾼다.

    입력:
    - delta_psnr = PSNR_FF - PSNR_OPT
    - tau: tie band. 보통 파일럿 변동성과 실용 최소 차이의 max.

    예상 결과:
    - feedforward_win / optimization_win / tie 중 하나를 반환한다.
    """

    if delta_psnr > tau:
        return "feedforward_win"
    if delta_psnr < -tau:
        return "optimization_win"
    return "tie"


def compute_tau(seed_variability: float, practical_min_delta: float = PRACTICAL_MIN_PSNR_DELTA) -> float:
    """승패 판정 임계값 tau를 계산한다.

    목적:
    - seed 변동성만으로 tau를 정하면 변동성이 큰 조건에서 tie가 과하게 넓어진다.
    - 그래서 파일럿 seed 변동성과 실용 최소 차이 중 큰 값을 사용한다.

    예상 결과:
    - 기본 practical_min_delta=0.5이면 파일럿 값이 0.2여도 tau=0.5가 된다.
    """

    return max(seed_variability, practical_min_delta)


def budget_checkpoint(checkpoints: Sequence[Mapping[str, float | str]], budget_seconds: float) -> Mapping[str, float | str] | None:
    """주어진 시간 예산 안에서 마지막 checkpoint를 선택한다.

    목적:
    - 메인 결과에서 test PSNR 최고 checkpoint를 고르는 oracle leakage를 막는다.
    - sparse-view optimization의 과적합 궤적을 그대로 관찰하기 위한 규칙이다.

    예상 결과:
    - wall_clock <= budget_seconds인 checkpoint 중 가장 늦은 것을 반환한다.
    - 예산 안에 checkpoint가 없으면 None을 반환한다.
    """

    eligible = [item for item in checkpoints if float(item["wall_clock"]) <= budget_seconds]
    if not eligible:
        return None
    return max(eligible, key=lambda item: float(item["wall_clock"]))


def oracle_checkpoint(checkpoints: Sequence[Mapping[str, float | str]], metric: str = "test_psnr") -> Mapping[str, float | str] | None:
    """진단용 oracle peak checkpoint를 선택한다.

    목적:
    - 메인 결과가 아니라 부록/분석용으로 `최고 가능 성능`을 따로 볼 때 사용한다.
    - test metric을 직접 보고 고르므로 main regime map에는 절대 쓰면 안 된다.

    예상 결과:
    - metric 값이 가장 큰 checkpoint를 반환한다.
    """

    if not checkpoints:
        return None
    return max(checkpoints, key=lambda item: float(item[metric]))


def scene_cluster_bootstrap_ci(
    rows: Sequence[Mapping[str, object]],
    value_fn: Callable[[Mapping[str, object]], float],
    scene_key: str = "scene",
    iterations: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """scene을 독립 단위로 보는 cluster bootstrap CI를 계산한다.

    목적:
    - 같은 scene의 seed 3회는 독립 표본이 아니라 반복 측정이다.
    - run 개수를 표본 수처럼 세면 confidence interval이 과하게 좁아진다.

    예상 결과:
    - mean: scene별 평균을 다시 평균낸 값.
    - ci_low / ci_high: scene cluster bootstrap 신뢰구간.
    - scene_count: 독립 단위로 사용한 scene 수.
    """

    # 먼저 scene별로 run을 묶고, scene 안 seed들은 평균 처리한다.
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[scene_key])].append(row)

    scenes = sorted(grouped)
    if not scenes:
        return {"mean": math.nan, "ci_low": math.nan, "ci_high": math.nan, "scene_count": 0}

    scene_values = {scene: mean(value_fn(row) for row in grouped[scene]) for scene in scenes}
    point_estimate = mean(scene_values.values())

    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        selected = [rng.choice(scenes) for _ in scenes]
        samples.append(mean(scene_values[scene] for scene in selected))

    alpha = 1.0 - confidence
    return {
        "mean": point_estimate,
        "ci_low": percentile(samples, alpha / 2.0),
        "ci_high": percentile(samples, 1.0 - alpha / 2.0),
        "scene_count": len(scenes),
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm 방식으로 다중 비교 보정 p-value를 계산한다.

    목적:
    - view 수, overlap, budget 조합별로 많은 비교가 생기므로 false positive를 줄인다.

    예상 결과:
    - 입력 p-value와 같은 순서의 adjusted p-value list를 반환한다.
    """

    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0 for _ in p_values]
    running_max = 0.0
    n = len(p_values)

    for rank, (original_index, p_value) in enumerate(indexed):
        corrected = min(1.0, (n - rank) * p_value)
        running_max = max(running_max, corrected)
        adjusted[original_index] = running_max

    return adjusted
