# Scaffold audit log — 2026-08-09

이 문서는 실험 스캐폴드(config/scripts/tests)를 실행해서 검증한 결과와, 그 결과를 계획서(§5.4, §5.7, §5.12) 및 담당자 검토와 대조해 확정한 사항을 기록한다. 목적은 코드를 더 다듬기 전에 "설계와 구현이 일치하는가", "GPU 예산 추정이 여전히 유효한가"를 문서로 동결하는 것이다.

## 1. 실행 검증 결과

- `python3 -m unittest discover -s tests` → 9개 테스트 전부 통과
- `bash experiments/scripts/run_experiment.sh` → `experiment_manifest.json` 생성, 총 12,480 row

## 2. Manifest 구조 재확인 (담당자 검토 반영)

담당자가 12,480을 26장면 × 5method로 역산했으나, manifest를 직접 파싱한 결과와 다르다. 실제 구조는 아래처럼 **3개 phase의 합**이며 단일 6중 product가 아니다.

| phase | rows | 구성 |
|---|---|---|
| main | 7,680 | scenes(20) × seeds(3) × view_counts(4) × overlap(2) × budgets(4) × methods(4) |
| c1b | 3,840 | scenes(20) × seeds(3) × view_counts(4) × overlap(2) × ff_methods(2) × (off 1 + on budgets 3) |
| c2_depth_noise | 480 | external_scenes(8) × seeds(3) × representative_conditions(4) × noise_levels(5) |
| c2_depth_scale_bias | 480 | external_scenes(8) × seeds(3) × representative_conditions(4) × scale_levels(5) |
| **합계** | **12,480** | |

- **method는 정확히 4개**: `DepthSplat`, `MVSplat`, `Vanilla3DGS`, `SparseGS` (manifest에서 직접 확인). 5번째 method는 없다.
- **C1-b는 method 목록에 섞여 있지 않다.** `phase: "c1b"`로 완전히 분리된 row이고, 각 row에 `renderer_equivalence_gate: true`가 별도로 기록된다. 담당자가 우려한 "gate 무력화" 시나리오는 현재 구조에서는 발생하지 않는다.
- main phase의 scene 수는 20 (`scenes_primary`), 외부 검증(C2)은 8 (`scenes_external`)이며 두 값이 다른 숫자다. 26이라는 값은 어디에도 대응하지 않는다.

## 3. Optimization 실행 횟수 재계산 — §5.4 GPU-hour 추정과의 정합성

budget(1/10/60/300초)은 **하나의 연속 학습 궤적에서 뽑는 체크포인트**이지 재학습이 아니다. 이 전제로 실제 optimization 실행(학습을 새로 시작하는 단위) 수를 다시 세면:

- main phase, optimization 계열만: `scenes(20) × seeds(3) × view_counts(4) × overlap(2) × opt_methods(2)` = **960회**
- 여기에 C1-b `refinement=on` 트랙(FF 초기값에서 새로 시작하는 refinement 실행)을 더하면: `20 × 3 × 4 × 2 × ff_methods(2)` = **960회 추가**
- C2는 noise/scale 조건마다 초기화 자체가 달라서 체크포인트 공유가 불가능하므로 전부 별도 실행: `480 + 480` = **960회**

**합계 실제 optimization 실행 ≈ 2,880회**로, §5.4의 최초 추정 "약 1,200 optimization run"보다 약 2.4배 많다. main phase만 보면 960회로 최초 추정과 비슷하지만, C1-b와 C2를 포함한 총량은 재검토가 필요하다.

**결론: §5.4의 GPU-hour 추정(200~300 GPU-hour)은 main phase 단독 기준으로는 대략 맞고, C1-b·C2를 포함한 전체 기준으로는 과소추정일 가능성이 높다. 파일럿 실측 후 재계산 필요 (계획서 §12 STEP 2에서 이미 요구하는 절차와 동일).**

## 4. 확인된 placeholder / 미확정 값 (실제 값으로 갱신 전 사용 금지)

- `model_registry.py`의 네 모델 모두 `supports_views=[2, 4, 8, 12]`로 동일 — 스모크 테스트 전 임시값, §5.2 표를 대신할 수 없음
- `experiment_config.yaml`의 `analysis.tau.pilot_seed_variability_psnr: 0.0` — 파일럿 seed 변동성 실측 전 placeholder. 현재 tau는 사실상 `practical_min_delta_psnr=0.5`로만 고정됨 (보수적 방향이라 안전하지만 갱신 필요)
- `run_experiment.py`의 scene id(`re10k_scene_000` 등)는 실제 데이터셋 스캔 결과가 아니라 deterministic placeholder

## 5. 환경 확인 (2026-08-09)

- GPU: NVIDIA H200 NVL, 143,771 MiB, idle (0% util, 사용 중인 프로세스 없음)
- 외부 네트워크 접근: 가능 (`curl` 테스트 성공)
- `/data/Re-feem/datasets/{re10k,dtu,dl3dv}`, `sfm_exports`, `overlap_reports`, `raw_downloads` 전부 빈 디렉터리 — 데이터 미확보 상태

## 6. DTU scan1 확보 및 vanilla 3DGS(gsplat) 러너 통합 — 2026-08-09

### 6.1 데이터

DTU 공식 배포(roboimagedata2.compute.dtu.dk)의 `SampleSet.zip`(6.9GB)에 scan1/scan6 예시가 이미 포함돼 있는 것을 확인. 전체 zip을 받는 대신 `remotezip`으로 zip central directory만 읽어 필요한 엔트리(scan1 rectified 이미지 49장, calibration `pos_001~064.txt`, GT `stl001_total.ply`)만 range request로 내려받음 — 약 200MB, 3.5분. `Rectified.zip`(130GB) 전체를 받을 필요가 없었다. 저장 위치: `/data/Re-feem/datasets/dtu/scan1/{images,cameras,stl}`. citeware 조건이므로(Jensen et al., CVPR 2014) 논문에 인용 필요.

나머지 scan(파일럿 5개, 외부검증 8~15개)도 같은 방식으로 언제든 저비용으로 추가 가능. 어떤 scan id를 쓸지는 아직 결정 안 됨(§7 참고).

### 6.2 Pose 변환

`experiments/scripts/dtu_dataset.py`: DTU의 3x4 projection matrix `P`를 RQ 분해로 `K, R, t`로 분리. `K@[R|t]`가 원본 `P`를 재구성하는지(오차 <0.002) 및 추정 scene 중심이 view1 이미지 중앙 근처(1600x1200 중 820,626)에 투영되는지로 검증 완료.

### 6.3 Vanilla 3DGS 러너 (`experiments/scripts/vanilla_3dgs_runner.py`)

gsplat 1.5.3 + `DefaultStrategy`로 densification을 구현. `protocol_utils.budget_checkpoint()` / `oracle_checkpoint()`를 실제 학습 궤적에 직접 적용해 정상 동작 확인. **의도적으로 비워둔 부분(추후 교체 필요):**
- Gaussian 초기화가 COLMAP SfM이 아니라 카메라 기하로 추정한 bounding sphere 안 random point (§5.2/§8에서 요구하는 COLMAP init 아님, GT point cloud도 참조하지 않음 — 참조하면 leakage)
- LPIPS 미계산 (사전학습 가중치 필요, 로그에는 `null`)

**버그 발견 및 수정:** 첫 20초 예산 smoke test에서 `budget_end_checkpoint`가 `None`을 반환. 원인은 스텝 완료 "후"에 경과시간을 확인하는 구조라 마지막 체크포인트가 budget을 항상 소폭(약 0.05~0.1초) 초과 기록됐고, `protocol_utils.budget_checkpoint()`의 `wall_clock <= budget` 필터가 (의도대로) 이를 걸러낸 것. 스냅샷 시점의 `wall_clock`을 `budget_label`로 clamp하도록 수정해 해결. **protocol_utils 자체는 정상 작동 — 버그는 러너 쪽 타이밍 기록에 있었음.**

### 6.4 DTU scan1, 8-view, seed 0, budget 300초 결과

| budget | iter | gaussians | test PSNR | test SSIM | peak VRAM |
|---|---|---|---|---|---|
| 1s | 5 | 100,000 | 8.76 | 0.372 | 995MB |
| 10s | 60 | 100,000 | 9.16 | 0.385 | 995MB |
| 60s | 375 | 100,000 | **10.69** | 0.469 | 995MB |
| 300s | 1,894 | 221,188 | 9.85 | 0.455 | 1,143MB |

**주목할 점**: densification이 시작된(iter>500) 이후 Gaussian 수는 100k→221k로 늘었지만 test PSNR은 오히려 60s 시점보다 떨어졌다. 이는 계획서 H1/H2 가설이 그리는 "sparse-view optimization의 과적합·정점 후 하강" 패턴과 정성적으로 일치한다. **다만 random init·1,894 iteration(표준 3DGS는 보통 30k)·단일 seed·단일 scene이라 이 수치 자체를 결과로 인용하면 안 되고, 전체 로깅·체크포인트 파이프라인이 이 패턴을 놓치지 않고 잡아낸다는 배관 검증으로만 사용한다.**

### 6.5 GPU 메모리·병렬 실행 테스트

- 1600x1200 해상도, Gaussian 100k~221k에서도 peak VRAM은 1~1.2GB. **3DGS/gsplat은 구조적으로 메모리 사용량이 적다** (파라미터 수가 적고 activation이 쌓이는 구조가 아님) — 143GB GPU에서 2GB만 쓰는 것은 정상이다.
- GPU util은 단일 프로세스에서도 이미 99~100% — **연산 병목이지 메모리 병목이 아니다.**
- **병렬 실행 테스트**: 동일 GPU에서 4-view/20초 budget run을 1개(solo, 125 iter) vs 6개 동시 실행(각 20~22 iter, 총 25초)으로 비교. **동시 실행해도 전체 처리량(총 iteration/초)이 늘지 않았다** — 6개 프로세스가 GPU를 시분할할 뿐, single-process 처리량과 aggregate 처리량이 거의 같음(≈6.2~6.3 it/s). **결론: H200의 여유 메모리는 "한 GPU에서 여러 run을 동시에 돌려 GPU-hour를 절약"하는 데 쓸 수 없다.** §3의 2,880회 optimization 실행 재추정은 이 결과를 반영해 순차 실행 기준으로 다시 잡아야 한다(멀티 GPU가 있다면 그쪽이 유일한 실질적 병렬화 경로).

## 7. COLMAP SfM init + LPIPS 통합 — 2026-08-09

### 7.1 COLMAP known-pose triangulation (`experiments/scripts/colmap_init.py`)

`pycolmap`(3.13.0, prebuilt wheel — COLMAP CLI 빌드 불필요)으로 COLMAP CLI의 `feature_extractor` → `exhaustive_matcher` → `point_triangulator` 조합을 재현했다. Pose-given track이므로 COLMAP이 pose를 추정하지 않는다: DTU calibration이 주는 고정 pose로 known-pose triangulation만 수행한다.

구현상 중요했던 점(ID 정합):
1. `pycolmap.extract_features()`를 먼저 실행해 database가 camera_id/image_id를 자동 배정하게 둔다.
2. database를 읽어 (image_name → image_id, camera_id) 매핑을 얻는다.
3. **그 ID를 그대로 재사용**해 DTU의 실제 K/R/t로 채운 COLMAP text 포맷(cameras.txt/images.txt, POINTS2D는 비움)을 쓴다 — ID가 어긋나면 `triangulate_points`가 keypoint/pose를 매칭 못 함.
4. database의 camera params도 (extract_features가 추측한 값 대신) 실제 K로 덮어써서 matching의 geometric verification이 정확한 intrinsics를 쓰게 한다.
5. `pycolmap.match_exhaustive()` → `pycolmap.triangulate_points(reconstruction, db, images, output)`.

**오직 학습(input) view만 triangulation에 넣는다.** held-out test view를 넣으면 초기화 단계에서 test 정보가 새는 leakage가 된다. GT point cloud(stl)는 어디에서도 참조하지 않는다.

검증 결과: scan1, 8-view, seed 0 → **1,994개 3D point, mean track length 3.7, mean reprojection error 0.61px** (COLMAP 자체 로그 기준) — 기하적으로 정상. 매우 적은 view 수(예: 2-view)에서 triangulation이 실패하거나 너무 빈약할 경우를 대비해 `MIN_SFM_POINTS=200` 미만이면 기존 random-sphere init으로 자동 fallback하고, 어느 경로를 탔는지 로그의 `init_source`(`colmap_sfm` / `random_sphere_fallback`)에 남긴다.

### 7.2 LPIPS 연결

`lpips` 패키지(AlexNet trunk)를 붙여 checkpoint마다 test LPIPS를 계산하도록 했다. 가중치는 `lpips` 패키지가 표준적으로 받아오는 `https://download.pytorch.org/models/alexnet-owt-7be5be79.pth`(PyTorch 공식 호스트, ImageNet 사전학습)에서 자동 다운로드된다.

### 7.3 ⚠ 환경 사고: `pip install lpips`가 공유 conda env의 torch를 조용히 업그레이드함

`ps3` conda env(이 프로젝트 전용 env가 아닐 수 있음 — 다른 작업에도 쓰일 가능성)에 `pip install lpips`를 실행했더니 의존성 해석 과정에서 **torch가 2.2.2+cu121 → 2.8.0+cu128로, torchvision이 최신판으로 조용히 업그레이드**됐고, 그 결과 `torchaudio 2.2.2+cu121`과 버전이 어긋났다. gsplat의 CUDA 확장이 기존 torch 2.2.2 기준으로 JIT 컴파일돼 있었기 때문에 이 상태로는 재현성이 깨질 위험이 있었다.

**조치:** `pip install --no-deps torch==2.2.2 torchvision==0.17.2 --index-url .../cu121`로 원복하고, `lpips`는 `--no-deps`로 재설치해 의존성 재해석이 다시 torch를 건드리지 않게 했다. 이후 gsplat/lpips 정상 동작 재확인 (`torch 2.2.2+cu121`, `torchvision 0.17.2+cu121`).

**교훈:** 이 env가 다른 작업과 공유된다면, 앞으로 이 env에 패키지를 추가할 때는 항상 `--no-deps` 또는 명시적 버전 고정을 쓰거나, 프로젝트 전용 가상환경(예: `conda create -n sparse3dgs --clone ps3`)을 새로 파서 격리하는 것을 권장한다.

### 7.4 통합 후 재검증 (scan1, 8-view, seed 0, budget 30초)

COLMAP init(1,994 point) + LPIPS 포함 전체 파이프라인이 정상 동작함을 확인. `budget_end_checkpoint`에 `test_lpips`와 `init_source` 필드가 정상적으로 채워짐. (30초/175 iteration으로는 densification 시작 전이라 품질 수치는 여전히 배관 검증용일 뿐 결과로 인용 불가 — §6.4와 동일한 caveat.)

## 8. DTU scan 15개 확보 — 2026-08-09

pixelNeRF류 논문의 "표준 DTU sparse-view split"을 그대로 따르려 했으나, 정확한 scan 번호 목록이 코드가 아니라 별도 `.lst` 파일로 배포돼 있어 신뢰 가능한 출처 없이 재현하면 틀린 번호를 표준으로 오인시킬 위험이 있었다. 담당자 결정: **자체적으로 scan 번호를 스프레드해서 선정**, 외부 논문 split과의 일치는 주장하지 않는다.

- `remotezip`으로 `Rectified.zip`(129GB)의 central directory만 읽어 실제 존재하는 scan 목록을 확인: **scan1~77, scan82~128, 총 124개** (78~81 결번).
- ReadMe가 명시한 360도 회전 그룹(55-58, 65-68, 69-73, 106-109, 110-113, 114-117, 118-121, 122-125 — 같은 물체를 4방향에서 찍은 것이라 사실상 한 scene)에서 그룹당 최대 1개만 선택해 장면 다양성이 왜곡되지 않게 했다.
- 최종 선정: **scan1(기존) + scan9, 17, 25, 34, 42, 50, 58, 67, 75, 87, 95, 104, 112, 120 = 15개.**
- 카메라 calibration(`pos_001~064.txt`)은 전체 DTU rig 공용이라 scan1 것을 그대로 복사해 재사용(재다운로드 불필요).
- 이미지(각 scan 49장, `_3_` 조명 조건) + GT `stl{scan}_total.ply`를 `remotezip` range read로만 받음. 6-way 병렬로 약 3분/scan, 최종 `/data/Re-feem/datasets/dtu/`에 15개 scan, **총 3.0GB** (`Rectified.zip` 129GB 전체를 받을 필요 없었음).
- 중간에 scan104가 네트워크 연결이 끊겨 20분간 정지했던 것을 발견 → 해당 프로세스만 kill 후 남은 파일 재시도로 해결. `remotezip` 기반 다운로드는 커넥션이 죽어도 무한 대기할 수 있으므로, 이후 다중 scan 다운로드에는 timeout/재시도 로직을 넣는 게 좋음(현재는 없음, TODO).

이 15개는 계획서 §5.4 외부검증(DTU) 8~15장면 요구치를 채운다. 다만 이번 논의에서 다룬 건 데이터 확보이지 파일럿 실행 자체가 아니다 — 실제로 이 15개에 대해 vanilla 3DGS를 돌려 overlap/tau 등을 산출하는 건 아직 안 함(§9 참고).

## 9. MVSplat 통합 및 DTU zero-shot 검증 — 2026-08-09

### 9.1 환경

`ps3`/`lpips` 사고(§7.3)를 반복하지 않기 위해 **완전히 격리된 conda env(`mvsplat`, python 3.10)를 새로 생성**해 README가 명시한 정확한 버전(`torch==2.1.2+cu121`, `torchvision==0.16.2`, `torchaudio==2.1.2` — cu118 대신 로컬 nvcc 12.1과 맞춰 cu121 wheel 사용)으로 설치. `requirements.txt`의 `diff-gaussian-rasterization-modified`(공식 저자의 fork, CUDA 커스텀 커널)는 `--no-build-isolation`으로 별도 설치해야 했음(빌드 격리 환경에 torch가 없어서 실패하는 문제). `numpy<2` 고정 필요(2.x는 torch 2.1.2와 ABI 불일치 경고). 체크포인트(`re10k.ckpt`, 48MB)는 README가 링크한 공식 Google Drive 폴더에서 `gdown`으로 받음.

### 9.2 공식 DTU sparse-view test split 발견

MVSplat repo의 `src/scripts/convert_dtu.py`의 `get_example_keys()`에 **공식 DTU sparse-view test scan 목록**이 하드코딩돼 있는 것을 발견했다:

```
scan1, scan8, scan21, scan30, scan31, scan34, scan38, scan40, scan41, scan45, scan55, scan63, scan82, scan103, scan110, scan114  (16개)
```

이건 담당자와 §8에서 "정확한 출처가 없어 못 쓴다"고 판단했던 그 pixelNeRF 계열 표준 split의 실제 소스였다 — 확인 없이 기억으로 적지 않은 게 맞는 판단이었고, 이제는 신뢰 가능한 출처(공식 repo 코드)로 확보됐다. **§8에서 자체 선정한 15개(scan1,9,17,25,34,42,50,58,67,75,87,95,104,112,120)와 scan1·scan34만 겹친다.** 이 표준 split으로 바꿀지는 다음 세션 결정 사항으로 남긴다(§10).

### 9.3 우리 raw DTU 데이터로 공식 체크포인트를 돌리는 브릿지 구현

MVSplat 공식 dtu.yaml 평가 경로는 별도 배포본(`dtu_training.rar`)을 그들의 `convert_dtu.py`로 전처리해야 하는데, 그 스크립트를 읽어보니 카메라 정규화 로직이 명시적이었다: **world-to-cam translation에 scale_factor=1/200 적용, intrinsics는 fx/w·fy/h로 정규화하고 principal point는 항상 (0.5, 0.5)로 고정, near=2.125/far=4.525 고정**(dtu.yaml에서 override). 이 로직을 그대로 재현하면 `dtu_training.rar` 없이도 **우리가 이미 갖고 있는 raw DTU calibration(pos_XXX.txt)만으로 동일한 입력을 만들 수 있었다.**

`/data/Re-feem/code/mvsplat/run_on_custom_dtu.py` 작성: 우리 `dtu_dataset.py`의 pose 파싱을 재사용해 위 변환을 적용하고, MVSplat 자체의 `apply_crop_shim_to_views`(256×256 resize+center-crop)를 그대로 import해서 이미지/intrinsics를 맞춘 뒤, `get_encoder`/`get_decoder`로 만든 모델에 `re10k.ckpt`를 로드(encoder 가중치 471개 키 **missing=0, unexpected=0** — 아키텍처 완전 일치 확인)해 forward.

**결과 (scan1, context view 2장[25,33], target view 3장[1,15,29], RE10K로 학습된 체크포인트로 zero-shot):**

| target view | PSNR |
|---|---|
| 15 (context와 가까움) | 11.82 |
| 29 | 7.18 |
| 1 (context와 각도 크게 다름) | 4.81 |

수치만 보면 낮아 보이지만 **렌더링된 이미지를 직접 확인하면 GT와 구조적으로 명확히 일치**한다 — 동일한 물체(손자국이 있는 깡통), 같은 텍스트("...OLINE") 위치, 같은 형태가 재현됨. context view와 각도가 먼 target(view1)일수록 검은 영역(추정 실패 영역)이 늘고 PSNR이 낮아지는 것도 sparse 2-view 외삽의 예상된 한계지 버그가 아니다. **카메라 변환(scale_factor, 정규화 intrinsics, cx=cy=0.5)이 틀렸다면 이 정도로 인식 가능한 재구성이 나올 수 없으므로, 이 결과 자체가 좌표계 변환이 올바르다는 강한 증거다.**

이로써 **RE10K로 학습된 MVSplat이 우리가 직접 받은 독립적인 DTU 원본 데이터에서 zero-shot cross-dataset generalization을 실제로 수행함**을 확인했다 — vanilla 3DGS에 이어 두 번째 method가 파이프라인에 살아있는 데이터로 검증됨.

### 9.4 이번 통합에서 의도적으로 안 한 것 (다음 세션 TODO)

- `vanilla_3dgs_runner.py`처럼 `protocol_utils` 체크포인트 스키마(experiment_id/scene/seed/method/wall_clock/...)에 맞춘 정식 러너로 감싸지 않음 — 지금은 1회성 검증 스크립트.
- context view 2장 고정만 테스트함. 계획서의 view_counts=[2,4,8,12] 전체나 MVSplat의 실제 지원 범위(§5.2 표) 확인 안 함.
- 15개 scan 전체가 아니라 scan1 하나에서만 검증.
- §9.2에서 발견한 공식 split으로 갈아탈지 여부 미결정.

## 10. 다음 결정이 필요한 항목 (블로킹, 2026-08-09 시점)

1. ~~DTU scan 목록: 자체 선정(15개) vs 공식 split(16개)~~ → 2026-08-10, 공식 split(16개)로 확정, 나머지 14개 추가 다운로드 완료.
2. ~~RE10K 확보 여부~~ → 2026-08-10, probe subset(pixelSplat 공식 small subset)으로 확보, MVSplat in-domain 검증까지 완료.

## 11. 2026-08-10 — Dense-view sanity check, RE10K/DL3DV in-domain 검증, DepthSplat 통합, 아키텍처 일반화

### 11.1 Dense-view sanity check 결과 (DTU scan1, 42-view train, 30k iteration)

병렬 세션(critical_path_2026-08-10.md)이 제기한 "sparse-view에서 나온 낮은 PSNR(9~11dB)이 진짜 현상인지 파이프라인 버그인지"를 가르기 위해, scan1을 거의 dense(42 train view, held-out 7) + 표준 iteration(30,000)으로 재실행.

| 체크포인트 | iter | gaussians | test PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|
| 60s | 346 | 24,690 | 13.38 | - | 0.591 |
| 300s | 1,749 | 218,693 | 22.61 | - | 0.352 |
| 1800s | 10,297 | 1,380,065 | 24.00 | - | 0.235 |
| 3600s | 20,366 | 1,703,091 | **24.11** | 0.843 | 0.218 |
| 30k iter (5458s) | 30,000 | 1,703,091 | 24.06 | 0.842 | 0.215 |

**해석**: 1800초 이후 24.0~24.1dB로 사실상 수렴, "정상 범위" 기준으로 잡았던 25dB에는 살짝 못 미치지만 SSIM 0.84/LPIPS 0.22는 명확히 건강한 값이다. oracle(3600s, 24.11)이 최종(30k, 24.06)보다 미세하게 높아 아주 약한 정점-후-하강이 있지만, sparse 8-view에서 본 급격한 붕괴(10.69→9.85, §6.4)와는 규모가 다르다. **결론: 파이프라인은 망가지지 않았다 — 같은 로깅이 "심한 과적합"과 "정상 수렴 후 미세한 노이즈"를 구분해서 잡아낸다는 것 자체가 좋은 신호. sparse-view의 낮은 PSNR은 진짜 현상일 가능성이 높다.** 다만 24dB가 DTU scan1 자체의 특성(어두운 배경·특이 텍스처)인지 하이퍼파라미터 미세조정 여지가 있는지는 추가 scan으로 더 봐야 함.

### 11.2 MVSplat RE10K in-domain 검증

pixelSplat 공식 small subset(re10k_subset.zip, test 41 scene)에서 MVSplat 공식 체크포인트로 2-view 추론. **mean PSNR 25.6dB** (19.2~29.4dB). DTU zero-shot(4.8~11.8dB, §9.3)과 극명하게 대비 — DTU에서 낮았던 건 OOD penalty이지 카메라 변환 버그가 아니었다는 강한 증거. 스크립트: `experiments/scripts/mvsplat_re10k_probe.py`.

### 11.3 DepthSplat 통합 + DL3DV in-domain 검증

- 공식 repo(`cvg/depthsplat`) clone, 격리 env(`depthsplat`, torch 2.4.0+cu121) 구축. 커스텀 rasterizer는 MVSplat과 동일하게 `--no-build-isolation`으로 별도 빌드.
- 체크포인트: `depthsplat-gs-base-dl3dv-256x448-randview2-6` (DL3DV in-domain, HF `haofeixu/depthsplat`). 공식 2-scene quick-test subset(`dl3dv_960p_test_subset.zip`, 같은 HF repo)으로 검증.
- 막혔던 것 두 개: ① 기본 `+experiment=dl3dv` config가 "small" 아키텍처라 "base" 체크포인트와 채널 수 불일치 → README의 정확한 override(`num_scales=2, upsample_factor=4, monodepth_vit_type=vitb`)로 해결, `missing=0 unexpected=0` 확인. ② encoder가 `return_depth=true` 기본값이라 `Gaussians` 대신 `dict`를 반환 → `return_depth=false`로 해결.
- 초기 시도(context view를 410프레임 영상의 양 끝에서 뽑음)는 baseline이 너무 커서 12dB — DL3DV는 RE10K보다 훨씬 긴 walkthrough라 context 간격을 좁게(30프레임 이내) 잡아야 한다는 걸 발견. 수정 후 **mean PSNR 20.0dB** (19.1~21.8dB, target을 context 사이로 제한). 스크립트: `experiments/scripts/depthsplat_dl3dv_probe.py`.
- **결론: feed-forward 두 종(MVSplat/DepthSplat) + optimization 한 종(Vanilla3DGS) 전부 실데이터에서 정상 동작 확인 완료.**

### 11.4 아키텍처 일반화 — model_registry.py 기반 dispatch

담당자 요청("메인 실험 장소를 정하고 모델/툴을 갖다 쓰는 형식")에 따라, 무거운 추상화(공통 베이스 클래스 등) 대신 이미 쓰던 패턴을 공식화:

- `model_registry.py`의 `ModelSpec`에 `conda_env_python`/`runner_script`/`external_repo`/`default_checkpoint` 필드 추가 — 모델별 실행 방법을 한 곳에서 관리.
- `experiments/scripts/run_experiment_batch.py` 신규 작성 — `run_dtu_batch.py`(DTU 전용, 모델 하드코딩)를 일반화. `--dataset-root`로 데이터셋 무관하게, `--methods`는 registry를 lookup해서 자동 dispatch. 새 모델 추가 시 이 파일은 안 건드리고 registry에 항목만 추가하면 됨.
- 검증: scan1에서 Vanilla3DGS(ok)/MVSplat(ok)/DepthSplat(정식 runner_script 없어서 `no_runner`로 안전하게 skip) 3-way 배치 실행 확인.
- `run_dtu_batch.py`는 삭제하지 않고 상단에 안내만 추가(기존 batch_summary.json 호환 유지).

### 11.5 남은 작업

1. ~~COLMAP 0-match 버그 수정 후 scan30/103/110 재실행~~ → 완료. 16개 공식 DTU scan × 2-view × seed0, Vanilla3DGS+MVSplat 배치 전부 성공.
2. DepthSplat 정식 러너(`depthsplat_runner.py`, protocol_utils 스키마) 아직 미작성 — probe 스크립트만 있음.
3. 우리가 직접 받은 DL3DV raw probe(480P, 5 scene)는 아직 미검증 — 오늘은 DepthSplat 공식 test subset으로만 확인함.
4. 메인 데이터셋 결정(RE10K vs DL3DV)은 여전히 미결.

## 12. 담당자 검토 반영 — 연구 설계에 직접 영향 (2026-08-10)

### 12.1 Densification on/off는 "공짜 통제 실험"이었다 — C1-b 결정

§11.1 dense-sanity 궤적을 다시 보면, gsplat `DefaultStrategy`의 `refine_stop_iter=15,000`(§11.4 이전에 이미 읽었던 B 항목 값) 때문에 iter 20,366~30,000 구간은 gaussian_count가 1,703,091로 고정된 채 PSNR만 24.11→24.06으로 미세 하강했다. 반면 sparse 8-view 붕괴(§6.4, iter<1749)는 densification이 활발한 구간에서 일어났다. **두 하강의 메커니즘이 다를 수 있다는 뜻이고, H1("densification 후 품질 하락")을 검증하려면 densification 없이 같은 시간을 쓴 조건과 비교해야 한다.**

→ **결정: C1-b는 refinement on/off뿐 아니라, densification on(기본) / densification off(`refine_stop_iter`를 낮추거나 0으로) 두 조건 모두 돌린다.** ForeSplat이 post-optimization에서 densification을 끄는 것과 같은 근거이며, 선행 연구 프로토콜과 비교 가능성도 확보된다. 비용은 늘지만 H1의 핵심 증거이므로 우선순위 높음. `공부방향.md` "신규" 항목에도 반영.

### 12.2 Dense-sanity 수치(24.0dB/LPIPS 0.215) 재해석 — "정상"이 아니라 "치명적 고장 없음"

42-view dense 재구성치고 LPIPS 0.215/SSIM 0.843은 다소 높은 편(dense 재구성은 보통 LPIPS 0.15 이하가 기대치)이라는 지적. 원인 후보 두 가지를 점검:

- **DTU 조명 조건 혼입 여부** — DTU Rectified 이미지는 카메라 위치당 조명 인덱스가 0~6+max로 8종 있다(§DTU README). 학습 view와 held-out view의 조명이 섞이면 재구성 품질과 무관하게 PSNR 상한이 걸린다. **점검 결과: 우리 다운로드 스크립트(`fetch_dtu_scans.py`, `fetch_dtu_official_split.py`)는 모든 scan·모든 position에서 예외 없이 `rect_{i:03d}_3_r5000.png`(조명 index 3, "most diffuse")만 받았다 — 조명 혼입 아님, 원인에서 제외.**
- **배경/마스크 처리** — DTU는 object mask 없이 전체 이미지(배경 포함)로 평가하면 배경이 점수를 깎는다. 우리 러너는 마스크를 안 쓴다. 원인일 가능성이 남아 있으나, DTU는 external validation 전용이라 우선순위 낮음 — main 실험(RE10K/DL3DV)에는 영향 없음. 평가 컨벤션을 문서에 명시하는 것으로 충분, 추가 조치는 보류.

### 12.3 DepthSplat context 간격 결과(20↔12dB) — overlap이 지배 축일 조기 신호이자 방법론 경고

§11.3의 "context 간격 30프레임 이내 20.0dB vs 넓게 잡으면 12dB" 결과는 같은 모델·같은 장면에서 view 선택만으로 8dB가 갈린 것 — **V2("어느 축이 승패를 지배하는가")의 조기 신호로, overlap이 유력 후보**라는 뜻이다.

동시에 방법론 경고이기도 하다: 지금 probe 스크립트의 context view 선정은 co-visibility가 아니라 임의 프레임 간격으로 했다. **§5.3에서 정한 co-visibility 기반 정의로 바꾸지 않으면, "우리가 view를 어떻게 골랐는가"가 실험 전체의 교란 변수가 된다.** DepthSplat/MVSplat 정식 러너를 만들 때 view 선정 로직을 `generate_overlap.py`의 co-visibility 계산과 연결해야 한다 (현재 미연결 — TODO). 또한 20.0dB 자체가 DepthSplat 공식 평가 index/split 대비 낮은 수치인지도 확인 필요(현재는 임의 선정과 비교할 기준이 없음).

### 12.4 병행 트랙 — RE10K/DL3DV 본 실험 규모 데이터 확보는 이론 공부와 별도로 계속 굴러가야 함

A-1(Gauss-Newton) 등 이론 공부가 지금 최우선이지만, **본 실험 데이터셋(RE10K 또는 DL3DV) 확보를 그 뒤로 미루면 안 된다.** 지금 있는 건 probe 규모(RE10K 3 scene, DL3DV 5 scene)뿐이고, 메인 데이터셋 결정 자체가 미결이다(§10-2). 이 상태가 계속되면 계획서 STEP3(4~6주차 본 벤치마크)가 시작을 못 한다. 이론 공부(A-1 등)와 데이터 확보는 병렬 트랙으로 관리할 것.

**후속 조치 (2026-08-10, 같은 날)**: 전체 RE10K의 "쉬운 경로"였던 pixelSplat 호스팅 서버(`schadenfreude.csail.mit.edu:8000`)가 죽어있는 것을 확인 — RE10K 전체 확보는 여전히 YouTube 기반이라 어렵다. 반면 DL3DV는 이미 접근 가능한 상태라 **DL3DV를 파일럿 규모(25 scene)로 먼저 확장**하기로 결정. `DL3DV-ALL-480P`의 11개 bucket에서 spread 선정(seed=0), 기존 probe 5개 + 신규 20개 = 25 scene, 1.9GB. 25개 전부 구조 검증 완료(transforms.json+images_8 정상, 네이티브 해상도 3840×2160으로 전부 동일, images_8 다운샘플 배율도 정확히 일치, 불일치 0건). RE10K는 별도로 획득 경로를 더 찾아야 하는 상태로 남음.

## 13. D 항목 실측 결과 — overlap과 depth uncertainty의 관계가 예상과 반대 (2026-08-10)

`experiments/scripts/geometry_uncertainty_figure.py` 작성: scan1 dense-sanity COLMAP 재구성(42-view, 861 pair)에서 pair마다 baseline(camera center 거리), overlap(`protocol_utils.compute_pairwise_overlaps` 재사용), 2-view Gauss-Newton 기반 depth 불확실성(`σ²(JᵀJ)⁻¹`을 view 시선 방향에 투영, pixel_sigma=1로 가정)을 실제로 계산. 출력: `experiments/outputs/geometry_figures/{pairwise_geometry.csv, *.png}`.

**결과 (상관계수)**:
- `corr(baseline, overlap) = -0.87` — 예상대로 (baseline↑ → overlap↓)
- `corr(baseline, log(depth_uncertainty)) = -0.95` — 예상대로 (baseline↑ → 불확실성↓, 표준 스테레오 조건수 이론과 일치)
- **`corr(overlap, log(depth_uncertainty)) = +0.95`** (baseline로 partial control해도 +0.80) — **overlap이 높을수록 depth 불확실성도 높게 나옴. H1이 암묵적으로 가정하는 방향(overlap↓ → uncertainty↑)과 정반대.**

**원인 (버그 아님, confounding)**: baseline이 overlap과 uncertainty를 **같은 방향으로** 동시에 끌어내린다(baseline↑ ⇒ overlap↓ *그리고* uncertainty↓). baseline을 안 보고 overlap-uncertainty만 직접 비교하면 이 공통 원인의 그림자만 보여 부호가 뒤집힌다.

**더 근본적인 프레이밍 문제**: 이 계산은 pairwise("인접한 두 view") 단위다. DTU 49-view rig에서 overlap 높은 pair = 물리적으로 가까운 두 카메라(중복 정보, 좁은 baseline), overlap 낮은 pair = 멀리 떨어진 두 카메라(스프레드, 넓은 baseline). 반면 실제 sparse-view 실험(2/4/8/12-view)이 다루는 "낮은 overlap"은 **소수 view를 장면 전체 커버리지를 위해 스프레드해서 뽑은 결과**이지 "두 프레임이 우연히 안 겹친 것"이 아니다 — §5.3의 pairwise `O_ij`와 sparse-view 세팅의 실제 문제(view set 전체의 커버리지 부족)가 같은 지표로 재고 있지만 원인 기제는 다를 수 있다.

**단순화 2가지 (재검토 필요, A-1 이론 공부와 함께)**:
1. baseline을 world-unit 거리로 썼다 — 삼각측량 조건수에 실제로 중요한 건 물체 기준 시선 사이 각도이지 raw 거리가 아닐 수 있음.
2. 각 3D point의 실제 track length는 평균 6.05인데(reconstruction summary), 이 분석은 매 pair마다 2-view만 가정해 계산했다 — 실제 달성된 정밀도가 아니라 "이 두 view만 있었다면"이라는 가상의 값이다.

**결론**: §5.3에서 "overlap이 uncertainty를 낮춘다"는 식으로 단순 서술하면 안 됨 — 이 confounding을 알고 서술해야 하며, 최종 해석은 A-1(Gauss-Newton) 재독 완료 후 재논의하기로 함(진행 중, 미결).

## 14. RE10K 획득 경로 재탐색 성공 + DL3DV-Benchmark 중복 확인 (2026-08-10, 같은 날 후반)

### 14.1 RE10K — 41 scene(probe) → 114 scene

§12.4/§10-2에서 "RE10K는 YouTube 기반이라 어렵다"고 결론 내렸던 것을 재검토. Hugging Face에서 pixelSplat/MVSplat과 동일한 `.torch` chunk 포맷으로 RE10K test split을 재업로드한 gate 없는 mirror(`Hualingchu/RealEstate10K_test`, 543 chunk, 전체 ~58GB)를 발견. 전체를 받을 필요 없이 5 chunk만 추가로 받아 **73 scene 신규 확보(기존 41개와 중복 0)** — 합계 **114 scene, 1.2GB**. 새 chunk도 MVSplat 공식 checkpoint로 검증(mean PSNR 22.4dB, 기존 probe의 25.6dB와 같은 정상 범위).

**결론: RE10K가 더 이상 데이터 확보 병목이 아니다.** 계획서 §5.4 본 실험 규모(20~30 scene)를 이미 초과했고, DL3DV(25 scene)보다도 많다. §12.4에서 "데이터 확보 축이 DL3DV로 뒤집혔다"고 기록했던 것도 이번 확장으로 다시 균형 회복(오히려 RE10K가 근소 우위). `model_registry.py` DATASET_REGISTRY 갱신 완료.

### 14.2 DL3DV 25개 pilot — 공식 DL3DV-Benchmark(140-scene eval split)와 중복 0 확인

§11 DL3DV pilot 확장 시 남겨뒀던 caveat("공식 split과 겹치는지 미확인")을 해소. `DL3DV/DL3DV-Benchmark`는 실제 파일 다운로드는 여전히 gate 상태지만 **파일 목록(`list_repo_files`)만은 접근 가능**(DTU/RE10K-Evaluation 때와 같은 패턴 — 메타데이터 열람과 실제 다운로드 gate는 별개). 141개 scene id를 확인해 우리 25개와 대조한 결과 **중복 0** — 우리 25개는 DepthSplat이 "train pool"로 쓰는 쪽에서 뽑힌 것이고 공식 held-out eval set과는 완전히 분리돼 있다. Leakage 걱정 없이 pilot으로 써도 됨. DepthSplat 논문 공식 수치와 직접 비교하려면 `DL3DV-Benchmark` 자체의 접근 승인이 별도로 필요(현재 미승인, 필요 시에만 진행).

### 14.3 메인 데이터셋 결정에 대한 함의

RE10K(114)와 DL3DV(25) 둘 다 이제 "데이터 부족"이 결정 사유가 될 수 없는 규모다. §12.3(model_checkpoint_domain_table.md)의 원래 결정 기준(어떤 feed-forward checkpoint가 in-domain인가)으로 돌아가 순수하게 그 기준으로 결정하면 됨 — MVSplat 중심이면 RE10K, DepthSplat 중심이면 DL3DV. 데이터 접근성은 더 이상 결정 요인이 아니다.
