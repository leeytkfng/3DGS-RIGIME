# 환경 복원 (2026-08-20)

황인재 서버에 `mvsplat`/`depthsplat`/`ps3`/`fsgs` conda 환경이 없어서 export한 파일들.
전부 CUDA 12.1 스택(Python 3.9/3.10, torch 2.1.2/2.2.2/2.4.0+cu121)이며, 4개 환경 모두
`pip list -e` 확인 결과 로컬 경로/editable install 없이 순수 conda+pip 패키지로만
구성되어 있어 이 export로 복원 가능하다.

## 복원 방법

```bash
# 1순위: conda export로 한 번에 (채널/버전까지 최대한 동일하게)
conda env create -n mvsplat    -f mvsplat_environment.yml
conda env create -n depthsplat -f depthsplat_environment.yml
conda env create -n fsgs       -f fsgs_environment.yml
conda env create -n ps3        -f ps3_environment.yml
```

`conda env create`가 채널/빌드 문제로 실패하면 순수 pip 방식으로 대체:

```bash
conda create -n mvsplat python=3.10 -y && conda activate mvsplat
pip install -r mvsplat_pip_freeze.txt
# depthsplat: python=3.10, fsgs: python=3.10, ps3: python=3.9 로 동일하게 반복
```

`*_environment.yml`은 `conda env export --no-builds`(machine-specific `prefix:` 줄 제거함)
결과이고, `*_pip_freeze.txt`는 해당 환경 안에서 `pip list --format=freeze`한 백업본이다.

## Git에 없던 입력 artifact (이번에 강제로 커밋함)

`experiments/outputs/`는 통째로 `.gitignore` 대상(체크포인트/로그가 100MB 넘는 게 흔해서)
이라 아래 4개 작은 JSON도 같이 제외되어 있었다 — 원래 데이터가 커서가 아니라 디렉터리
전체가 걸려서였다. `overlap 본 실험` runner 두 개(`run_re10k_overlap_supplement.py`,
`run_dl3dv_overlap_supplement.py`)가 기본값으로 참조하는 파일이 정확히 이 4개뿐이라
`git add -f`로 강제 추가했다:

- `experiments/outputs/re10k_main_subset/re10k_main_subset.json` (RE10K 30-scene 서브셋)
- `experiments/outputs/re10k_overlap_candidates/re10k_overlap_candidates.json` (RE10K overlap candidates)
- `experiments/outputs/dl3dv_overlap_lowhigh/dl3dv_overlap_candidates.json` (DL3DV overlap candidates)
- `experiments/outputs/dl3dv_overlap_v2/all_scenes_summary.json` (DL3DV overlap summary)

각 파일은 scene ID·view index·overlap 점수 메타데이터일 뿐 이미지/체크포인트는 없어
전부 130KB 이하다. `git pull`만 받으면 runner가 그대로 돌아간다 — 별도 전달 불필요.

같은 폴더 안의 `colmap_work/` 서브디렉터리(수백MB)는 이 4개 JSON을 만들 때 쓴 중간
산물일 뿐, overlap 본 실험 runner는 실행 시점에 COLMAP triangulation을 scene마다 새로
돌리므로(`vanilla_3dgs_runner.py`) 이 중간 산물은 필요 없다 — 전달 대상에서 뺐다.
