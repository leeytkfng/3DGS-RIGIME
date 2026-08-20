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

## ⚠ `diff-gaussian-rasterization` / `simple-knn`은 위 export에 못 담았다 — 별도 빌드 필요

이 둘은 PyPI에 없는 CUDA 확장 패키지라 `pip==0.0.0`으로만 잡히고(설치 시 리졸브 불가),
`*_environment.yml`/`*_pip_freeze.txt`에서도 해당 줄을 지워놨다. 대신 실제 설치에 쓰인
소스를 `direct_url.json`(pip가 VCS/로컬 설치 시 기록하는 정확한 origin) 기준으로 역추적
완료 — 아래 명령으로 conda env 만든 뒤 이어서 각각 빌드하면 된다.

### mvsplat, depthsplat (동일 소스)

MVSplat/DepthSplat 계열은 pixelSplat 저자(dcharatan)의 수정판을 쓴다(원본 3DGS 저장소가
아님 — antialiasing/depth-output/confidence 없는 더 단순한 버전). `requirements.txt`에
git URL만 적혀 있고 커밋이 안 박혀 있어서, 실제 설치된 커밋을 `direct_url.json`으로
확인했다:

```bash
conda activate mvsplat   # 또는 depthsplat — 먼저 torch부터 environment.yml로 설치돼 있어야 함
pip install "git+https://github.com/dcharatan/diff-gaussian-rasterization-modified@1250c420ebb945f0dce9945086e22faab9157c92"
```

두 환경 모두 정확히 이 커밋(`1250c420ebb945f0dce9945086e22faab9157c92`)이었다. nvcc가 활성
CUDA 12.1 툴체인을 가리키고 있어야 컴파일된다(torch는 mvsplat=2.1.2+cu121,
depthsplat=2.4.0+cu121 — environment.yml에 이미 포함).

### fsgs (FSGS 자체 confidence 변형 + simple-knn)

FSGS는 `.gitmodules`에 원본 graphdeco-inria 저장소를 submodule로 선언해놓고 있지만,
**실제 빌드에 쓰인 소스는 FSGS 저장소 자체에 일반 파일로 커밋되어 있는
`submodules/diff-gaussian-rasterization-confidence`다**(별도 서브모듈 URL이 아니라
FSGS 저장소를 클론하면 바로 딸려 옴 — 논문 §V.4/§Confidence 서술의 근거인 FSGS의
`confidence` 필드가 바로 이 변형에서 나온다). `submodules/simple-knn`도 마찬가지로
FSGS 저장소에 직접 커밋되어 있다.

```bash
conda activate fsgs
git clone https://github.com/VITA-Group/FSGS.git /tmp/FSGS_src
cd /tmp/FSGS_src && git checkout a536a64c5b366b1088be64eeadf9e791ca26897c
pip install submodules/diff-gaussian-rasterization-confidence
pip install submodules/simple-knn
```

(`git submodule update --init`은 필요 없다 — 이 두 디렉터리는 진짜 submodule이 아니라
FSGS 저장소에 직접 커밋된 일반 디렉터리다. `.gitmodules`의 선언과 실제 내용이 다르다는
점을 클론 직후 `git ls-tree HEAD submodules/`로 한 번 확인해보길 권한다.)

### ps3

`ps3` 환경은 `diff-gaussian-rasterization`을 아예 안 쓴다 — 이 프로젝트의 Vanilla3DGS는
gsplat(`pip install gsplat`, PyPI에 있음, 논문 §III.3 "3DGS~\cite{kerbl20233d} ...
gsplat~\cite{ye2024gsplat} 구현" 참고) 기반이라 위 빌드가 필요 없다.

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
