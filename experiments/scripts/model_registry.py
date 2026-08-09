#!/usr/bin/env python3
"""실험 scaffold에서 사용할 모델/데이터셋 registry.

이 파일의 목적:
- config에 적힌 method/dataset 이름이 어떤 의미인지 한 곳에서 관리한다.
- run_experiment.py가 manifest를 만들 때 method family, pose 필요 여부, 지원 view 수를 함께 기록한다.
- 예상 결과는 manifest의 `models`, `datasets` 섹션에 들어가는 metadata다.
"""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ModelSpec:
    """실험에 등록된 모델 하나의 기본 정보.

    필드 의미:
    - name: config와 manifest에서 사용할 모델 이름.
    - family: feedforward 또는 optimization 계열 구분.
    - requires_pose: pose-given track에서 pose 입력이 필요한지 여부.
    - supports_views: smoke test 전 임시 지원 view 목록. 실제 모델 확인 후 수정한다.
    - notes: 재현 상태나 구현 연결 시 주의사항.
    """

    name: str
    family: str  # feedforward or optimization
    requires_pose: bool
    supports_views: List[int]
    notes: str = ""


# 모델 registry.
# 목적: config에서 method 이름만 쓰고, 세부 metadata는 이곳에서 가져간다.
# 예상 결과: run_experiment.py가 이 정보를 manifest에 복사해 실험 당시 모델 가정을 남긴다.
MODEL_REGISTRY = {
    "DepthSplat": ModelSpec(
        name="DepthSplat",
        family="feedforward",
        requires_pose=True,
        supports_views=[2, 4, 8, 12],
        notes="Zero-shot feed-forward model. 실제 checkpoint와 view 지원 범위는 smoke test 후 갱신.",
    ),
    "MVSplat": ModelSpec(
        name="MVSplat",
        family="feedforward",
        requires_pose=True,
        supports_views=[2, 4, 8, 12],
        notes="Zero-shot feed-forward model. 실제 입력 해상도와 view 수 제약은 smoke test 후 갱신.",
    ),
    "Vanilla3DGS": ModelSpec(
        name="Vanilla3DGS",
        family="optimization",
        requires_pose=True,
        supports_views=[2, 4, 8, 12],
        notes="Standard 3DGS optimization baseline. budget checkpoint 로그 연결 필요.",
    ),
    "SparseGS": ModelSpec(
        name="SparseGS",
        family="optimization",
        requires_pose=True,
        supports_views=[2, 4, 8, 12],
        notes="Sparse-view specialized optimization baseline. 코드 안정성 확인 후 FSGS와 교체 가능.",
    ),
}


# 데이터셋 registry.
# 목적: 데이터 루트가 바뀌어도 config/manifest가 같은 dataset 이름을 쓰도록 한다.
# 예상 결과: manifest에 resolved_path가 기록되어 결과 재현 시 어떤 경로를 썼는지 알 수 있다.
DATASET_REGISTRY = {
    "RE10K": {
        "path": "/data/Re-feem/datasets/re10k",
        "description": "Main benchmark candidate for large-scale sparse-view experiments.",
        "recommended_scenes": 20,
        "notes": "Good for primary regime-map experiments and diverse indoor/outdoor scenes.",
    },
    "DL3DV": {
        "path": "/data/Re-feem/datasets/dl3dv",
        "description": "High-quality multi-view dataset with richer geometry variation.",
        "recommended_scenes": 20,
        "notes": "Useful as a secondary benchmark if available.",
    },
    "DTU": {
        "path": "/data/Re-feem/datasets/dtu",
        "description": "External validation set with GT geometry available.",
        "recommended_scenes": 8,
        "notes": "Best choice for depth/geometry and failure analysis.",
    },
}


def get_model_spec(name: str) -> ModelSpec:
    """모델 이름으로 ModelSpec을 가져온다.

    목적:
    - runner 연결 시 method 이름 오타를 빠르게 잡는다.

    예상 결과:
    - 등록된 이름이면 ModelSpec 반환.
    - 등록되지 않은 이름이면 KeyError 발생.
    """

    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model: {name}")
    return MODEL_REGISTRY[name]


def get_dataset_spec(name: str) -> dict:
    """데이터셋 이름으로 경로/설명 정보를 가져온다.

    목적:
    - dataset 이름과 실제 저장 경로를 분리해 관리한다.

    예상 결과:
    - 등록된 이름이면 dict 반환.
    - 등록되지 않은 이름이면 KeyError 발생.
    """

    if name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset: {name}")
    return DATASET_REGISTRY[name]
