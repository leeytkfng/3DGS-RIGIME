#!/usr/bin/env python3
"""Co-visibility 기반 view selector — overlap_levels(고/저) 축을 실제로 통제하기 위한 선택 로직.

배경(2026-08-13): `run_experiment.py`의 manifest 생성은 `product(scenes, seeds, view_counts,
overlap_levels, budgets, methods)`로 되어 있어, **같은 scene·view_count에서 overlap=low와
overlap=high 후보가 둘 다 있어야 한다**고 이미 전제한다. 그런데 `generate_overlap.py`류는
overlap을 "측정"만 할 뿐 "선택"하지 않는다 — RE10K는 순수 랜덤(view_count>2), DL3DV v2는
DepthSplat의 실제 알고리즘(좁은 window 안 farthest-point sampling, `view_sampler_bounded_v2.py`
재현)을 쓰지만 둘 다 후보를 1개만 만든다. 이 모듈이 그 공백을 메운다.

원리 — camera 위치(3D center)에 대한 farthest-point sampling(FPS)을 두 가지 다른 pool에 적용:
- **high overlap 후보**: 전체 pool 중 좁은 window(연속된 구간) 안에서만 FPS. window이 좁으니
  뽑힌 view들도 서로 가깝다 -> baseline이 짧아 co-visibility가 높다. (DL3DV v2가 이미
  DepthSplat 재현으로 이렇게 하고 있었다 — 이 모듈은 그 로직을 일반화해 재사용한다.)
- **low overlap 후보**: 전체 pool 전체 범위에서 FPS. 넓게 퍼진 view들이라 baseline이 길어
  co-visibility가 낮다.

같은 FPS 알고리즘을 pool만 바꿔 쓰므로, 두 후보의 차이는 순수하게 "얼마나 넓은 범위에서
뽑았는가"뿐이다 — 다른 교란요인(알고리즘 종류, 무작위성 정도)이 섞이지 않는다.

**주의(2026-08-13)**: 이 selector가 생성하는 후보는 기존 `re10k_main_subset.json`/
`dl3dv_overlap_v2/`의 view 선택과 다르다(같은 scene이라도 다른 view가 뽑힐 수 있음).
이번 세션에 이미 쌓인 실측 결과(C1-a seed×3 파일럿, FSGS 검증 등)는 전부 기존 후보를 썼으므로,
재현성 보존을 위해 기존 파일은 건드리지 않고 이 selector의 결과는 별도 파일로 남긴다
(`generate_re10k_overlap_candidates.py`/`generate_dl3dv_overlap_candidates.py` 참고).
"""

from __future__ import annotations

import numpy as np


def farthest_point_sample(positions: np.ndarray, npoint: int, seed: int = 0) -> list[int]:
    """`generate_dl3dv_view_overlap.py`의 동명 함수를 공용 모듈로 승격한 것 — DepthSplat
    `view_sampler_bounded_v2.py::farthest_point_sample()`의 numpy 재현과 동일 알고리즘.
    centroid에서 가장 먼 점부터 시작해 매번 이미 뽑힌 집합에서 가장 먼 점을 추가한다.

    원본은 시작점이 `np.random.randint`(전역 상태 의존)였는데, 여기서는 `seed`로 결정론적으로
    만든다 — low/high 후보를 같은 scene에서 재현 가능하게 여러 번 만들어야 하기 때문이다.
    """

    n = positions.shape[0]
    if npoint >= n:
        return list(range(n))

    rng = np.random.default_rng(seed)
    distance = np.full(n, 1e10)
    centroid = positions.mean(axis=0)
    # 원본은 centroid에서 가장 먼 점으로 결정론적으로 시작하지만, low/high 두 후보가 항상
    # 같은 시작점을 쓰면 다양성이 줄어드니 seed로 살짝 흔든다(centroid 기준 상위 25% 중 무작위).
    dist_to_centroid = np.sum((positions - centroid) ** 2, axis=1)
    top_quartile = max(1, n // 4)
    candidates = np.argsort(-dist_to_centroid)[:top_quartile]
    farthest = int(rng.choice(candidates))

    selected: list[int] = []
    for _ in range(npoint):
        selected.append(farthest)
        dist = np.sum((positions - positions[farthest]) ** 2, axis=1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = int(np.argmax(distance))
    return selected


def select_high_overlap_indices(
    pool_indices: list[int],
    positions: np.ndarray,
    view_count: int,
    seed: int,
    window_multiplier: float = 4.0,
) -> list[int]:
    """`pool_indices`(예: test 제외한 frame index 목록)와 그 위치(`positions`, pool과 같은 순서)
    중 좁은 window 안에서 FPS로 view_count개를 뽑는다 — 서로 가까운 view들 = high overlap 목표.

    window_multiplier: window 크기를 view_count의 몇 배로 잡을지(기본 4배 — DL3DV v2가 쓰던
    MAX_CONTEXT_GAP=50과 대략 비슷한 비율, view_count=12일 때 48).
    """

    n = len(pool_indices)
    window_size = min(n, max(view_count, int(view_count * window_multiplier)))

    rng = np.random.default_rng(seed)
    window_start = int(rng.integers(0, n - window_size + 1)) if n > window_size else 0
    window_local = list(range(window_start, window_start + window_size))

    window_positions = positions[window_local]
    fps_local = farthest_point_sample(window_positions, view_count, seed=seed)
    return sorted(pool_indices[window_local[i]] for i in fps_local)


def select_low_overlap_indices(
    pool_indices: list[int],
    positions: np.ndarray,
    view_count: int,
    seed: int,
) -> list[int]:
    """`pool_indices` 전체 범위에서 FPS로 view_count개를 뽑는다 — 넓게 퍼진 view들 = low overlap 목표."""

    fps_local = farthest_point_sample(positions, view_count, seed=seed)
    return sorted(pool_indices[i] for i in fps_local)


def select_overlap_candidates(
    pool_indices: list[int],
    positions: np.ndarray,
    view_count: int,
    seed: int,
    window_multiplier: float = 4.0,
) -> dict[str, list[int]]:
    """high/low 후보를 한 번에 만든다. `pool_indices`/`positions`는 같은 순서로 대응해야 한다."""

    return {
        "high": select_high_overlap_indices(pool_indices, positions, view_count, seed, window_multiplier),
        "low": select_low_overlap_indices(pool_indices, positions, view_count, seed),
    }
