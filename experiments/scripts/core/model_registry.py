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
    """실험에 등록된 모델 하나의 기본 정보이자, run_dtu_batch.py류 dispatcher가
    "이 모델을 어떤 env의 어떤 스크립트로 돌리는가"를 찾는 lookup 지점.

    필드 의미:
    - name: config와 manifest에서 사용할 모델 이름.
    - family: feedforward 또는 optimization 계열 구분.
    - requires_pose: pose-given track에서 pose 입력이 필요한지 여부.
    - supports_views: smoke test 전 임시 지원 view 목록. 실제 모델 확인 후 수정한다
      (2026-08-10 기준 실제로 검증된 건 MVSplat/DepthSplat 2-view뿐 — 나머지는 아직 placeholder).
    - conda_env_python: 이 모델을 실행하는 격리된 conda env의 python 바이너리 절대경로.
      서로 다른 모델이 서로 다른 torch 버전을 요구해서 env를 공유하면 안 된다(§7.3/§9.1 audit log 사고).
      None이면 아직 env를 안 만들었거나(SparseGS) 우리 repo의 기존 env(ps3)로 충분한 경우.
    - runner_script: protocol_utils 로깅 스키마를 따르는 정식 러너 스크립트 경로(repo 기준 상대경로).
      None이면 아직 정식 러너가 없고 1회성 probe 스크립트만 있는 상태.
    - external_repo: 모델 코드가 사는 외부 clone 위치. None이면 우리 repo에 자체 구현됨(Vanilla3DGS).
    - default_checkpoint: 기본으로 쓰는 사전학습 체크포인트 경로. optimization 계열은 None.
    - notes: 재현 상태나 구현 연결 시 주의사항.
    """

    name: str
    family: str  # feedforward or optimization
    requires_pose: bool
    supports_views: List[int]
    conda_env_python: str | None = None
    runner_script: str | None = None
    external_repo: str | None = None
    default_checkpoint: str | None = None
    notes: str = ""


# 모델 registry.
# 목적: config에서 method 이름만 쓰고, 세부 metadata는 이곳에서 가져간다.
# 예상 결과: run_experiment.py가 이 정보를 manifest에 복사해 실험 당시 모델 가정을 남긴다.
# dispatcher(run_dtu_batch.py 등)는 conda_env_python + runner_script만 보고 subprocess를 띄우면 된다.
MODEL_REGISTRY = {
    "DepthSplat": ModelSpec(
        name="DepthSplat",
        family="feedforward",
        requires_pose=True,
        supports_views=[2, 4],  # 체크포인트 randview2-6 학습 분포 기준(§5.2, 2026-08-12). 8/12는 OOD.
        conda_env_python="/opt/conda/envs/depthsplat/bin/python3",
        runner_script=None,  # 아직 probe 스크립트뿐: experiments/scripts/probes/depthsplat_dl3dv_probe.py
        external_repo="/data/Re-feem/code/depthsplat",
        default_checkpoint="/data/Re-feem/code/depthsplat/pretrained/depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth",
        notes="2026-08-10: DL3DV in-domain probe로 mean PSNR 20.0dB 확인(2-view, 공식 test subset). "
        "2026-08-12: README/config 확인 결과 이 체크포인트는 2~6-view 랜덤 샘플링으로 학습됨(randview2-6) — "
        "supports_views를 [2,4]로 좁힘(6은 우리 view_counts 축에 없어 제외, 8/12는 분포 밖). "
        "공식으로는 별도 체크포인트(randview4-10, 448x768)로 최대 12-view까지 지원하나 아직 미다운로드. "
        "4/8/12-view 실측 및 정식 러너(protocol_utils 스키마)는 아직 미구현.",
    ),
    "MVSplat": ModelSpec(
        name="MVSplat",
        family="feedforward",
        requires_pose=True,
        supports_views=[2],  # RE10K 학습 분포는 고정 2-view(§5.2, 2026-08-12). 4/8/12는 저자도 OOD로 간주.
        conda_env_python="/opt/conda/envs/mvsplat/bin/python3",
        runner_script="experiments/scripts/runners/mvsplat_runner.py",
        external_repo="/data/Re-feem/code/mvsplat",
        default_checkpoint="/data/Re-feem/code/mvsplat/checkpoints/re10k.ckpt",
        notes="2026-08-10: DTU zero-shot(2-view) 검증 + RE10K in-domain probe mean PSNR 25.6dB 확인. "
        "2026-08-12: config(`view_sampler/bounded.yaml`)가 RE10K를 고정 2-view로 학습시킴을 확인, "
        "DTU 공식 eval index도 N=2,3만 제공. README가 직접 '12-view는 DepthSplat 쓰라'고 안내 — "
        "supports_views를 [2]로 좁힘. DTU smoke(2026-08-11)에서 4/8/12-view forward pass 자체는 안 죽었지만 "
        "이는 분포 밖 사용이지 공식 지원이 아님(§5.2).",
    ),
    "Vanilla3DGS": ModelSpec(
        name="Vanilla3DGS",
        family="optimization",
        requires_pose=True,
        supports_views=[2, 4, 8, 12],
        conda_env_python="/opt/conda/envs/ps3/bin/python3",
        runner_script="experiments/scripts/runners/vanilla_3dgs_runner.py",
        external_repo=None,  # gsplat pip 패키지 + 자체 구현(COLMAP init 포함), 외부 clone 없음
        default_checkpoint=None,
        notes="2026-08-10: DTU 16-scan 배치(2-view) + 42-view/30k-iter dense sanity check(24dB대) 검증 완료.",
    ),
    "FSGS": ModelSpec(
        name="FSGS",
        family="optimization",
        requires_pose=True,
        supports_views=[2, 4, 8, 12],
        conda_env_python="/opt/conda/envs/fsgs/bin/python3",
        runner_script="experiments/scripts/runners/fsgs_runner.py",
        external_repo="/data/Re-feem/code/fsgs",
        default_checkpoint=None,
        notes="Sparse-view specialized optimization baseline, SparseGS 대신 채택(2026-08-12, 코드 상태/의존성 기준). "
        "2026-08-13: `fsgs_runner.py` 작성 — FSGS 자체 Scene/GaussianModel/render()/loss는 그대로 재사용하되 "
        "(a) view 선택을 우리 seed 기반 train_ids/test_ids로 monkey-patch(원본 readColmapSceneInfo의 "
        "llffhold+linspace는 우리 선택을 무시하는 버그성 불일치였음), (b) 바깥 루프를 iteration 기반에서 "
        "wall-clock budget 기반으로 교체. 초기화는 dense-MVS 대신 Vanilla3DGS와 동일한 sparse COLMAP "
        "triangulation(overall.md §4.2에 명시적 프로토콜 편차로 문서화). 현재 DTU만 지원, RE10K/DL3DV는 "
        "다음 단계. 실제 학습 검증은 DTU scan1/seed0/12-view/1000-iter로 완료(구 prep 스크립트 기준, "
        "새 runner로 재검증 필요).",
    ),
}


# 데이터셋 registry.
# 목적: 데이터 루트가 바뀌어도 config/manifest가 같은 dataset 이름을 쓰도록 한다.
# 예상 결과: manifest에 resolved_path가 기록되어 결과 재현 시 어떤 경로를 썼는지 알 수 있다.
DATASET_REGISTRY = {
    "RE10K": {
        "path": "/data/Re-feem/datasets/re10k",
        "description": "Main benchmark candidate for large-scale sparse-view experiments.",
        "recommended_scenes": 114,
        "notes": "2026-08-10: pixelSplat probe(41 scene) + HF mirror `Hualingchu/RealEstate10K_test`"
        "(gate 없음, 543 chunk 중 5개)에서 73 scene 추가 확보, 합계 114 scene(1.2GB), 중복 0. "
        "test split만 있음(train 없음). MVSplat in-domain 검증(새 chunk 기준 mean PSNR 22.4dB)까지 확인.",
    },
    "DL3DV": {
        "path": "/data/Re-feem/datasets/dl3dv",
        "description": "High-quality multi-view dataset with richer geometry variation.",
        "recommended_scenes": 25,
        "notes": "2026-08-10: 25 scene 확보(pilot 규모, DL3DV-ALL-480P에서 bucket별 spread 선정). "
        "DepthSplat in-domain 검증(mean PSNR 20.0dB)은 별도 공식 test subset으로 완료, 이 25개 자체는 아직 미검증.",
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
