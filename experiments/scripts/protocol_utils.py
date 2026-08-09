#!/usr/bin/env python3
"""Protocol helpers for the sparse-view 3DGS regime study scaffold."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations
from statistics import mean
from typing import Callable, Iterable, Mapping, Sequence


PRACTICAL_MIN_PSNR_DELTA = 0.5


@dataclass(frozen=True)
class PairwiseOverlap:
    view_i: str
    view_j: str
    overlap: float
    shared_points: int
    points_i: int
    points_j: int
    connected: bool

    def to_dict(self) -> dict[str, float | int | str | bool]:
        return asdict(self)


def compute_pairwise_overlaps(
    view_points: Mapping[str, Iterable[str | int]],
    min_common_points: int = 1,
) -> list[PairwiseOverlap]:
    """Compute pairwise co-visibility, keeping failed/non-edge pairs as zero."""

    point_sets = {view: set(points) for view, points in view_points.items()}
    overlaps: list[PairwiseOverlap] = []

    for view_i, view_j in combinations(sorted(point_sets), 2):
        points_i = point_sets[view_i]
        points_j = point_sets[view_j]
        shared = len(points_i & points_j)
        denominator = len(points_i) + len(points_j)

        # Paper-critical rule: failed matches and too-weak shared tracks are not
        # dropped. They are evidence of low overlap, so they must enter as zero.
        if shared < min_common_points or denominator == 0:
            overlap = 0.0
            connected = False
        else:
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
    """Aggregate overlap with zero-included mean as the primary statistic."""

    pairwise = compute_pairwise_overlaps(view_points, min_common_points=min_common_points)
    values = [item.overlap for item in pairwise]
    zero_pairs = sum(1 for item in pairwise if item.overlap == 0.0)

    # Mean over all pairs is the regime-map x-axis. Median is kept only as a
    # reference because it can hide many zero-overlap pairs in sparse settings.
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
    pairwise = compute_pairwise_overlaps(view_points, min_common_points=min_common_points)
    return {
        "summary": aggregate_overlap(view_points, min_common_points=min_common_points),
        "pairwise": [item.to_dict() for item in pairwise],
    }


def classify_delta(delta_psnr: float, tau: float) -> str:
    if delta_psnr > tau:
        return "feedforward_win"
    if delta_psnr < -tau:
        return "optimization_win"
    return "tie"


def compute_tau(seed_variability: float, practical_min_delta: float = PRACTICAL_MIN_PSNR_DELTA) -> float:
    # The practical floor prevents noisy pilot seeds from being the only source
    # of the tie band used in the final win/loss regime map.
    return max(seed_variability, practical_min_delta)


def budget_checkpoint(checkpoints: Sequence[Mapping[str, float | str]], budget_seconds: float) -> Mapping[str, float | str] | None:
    """Return the last checkpoint at or before the budget without oracle selection."""

    # Main results must use this fixed time rule. Selecting the best test PSNR
    # would leak test information and erase the overfitting signal we study.
    eligible = [item for item in checkpoints if float(item["wall_clock"]) <= budget_seconds]
    if not eligible:
        return None
    return max(eligible, key=lambda item: float(item["wall_clock"]))


def oracle_checkpoint(checkpoints: Sequence[Mapping[str, float | str]], metric: str = "test_psnr") -> Mapping[str, float | str] | None:
    """Separate diagnostic helper; never use for the main regime-map result."""

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
    """Bootstrap confidence interval using scene as the independent unit."""

    # Runs from different seeds in the same scene are repeated measurements, not
    # independent samples. Group first, then resample scenes as clusters.
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
    """Holm adjusted p-values in the original input order."""

    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0 for _ in p_values]
    running_max = 0.0
    n = len(p_values)

    for rank, (original_index, p_value) in enumerate(indexed):
        corrected = min(1.0, (n - rank) * p_value)
        running_max = max(running_max, corrected)
        adjusted[original_index] = running_max

    return adjusted
