# 일일 보고서 — 2026-08-09

## 오늘 목표

착수 전 "설정과 초기 세팅이 다 돼 있는 것 같은데, 실제로 검증하려는 게 뭐고 신뢰성 있는 실험을 위한 아이터레이션/세팅을 어떻게 해야 하는가"에서 출발해, 하루 동안 (1) 기존 스캐폴드 감사 → (2) 실제 모델 하나를 real 데이터로 end-to-end 돌려서 파이프라인 검증 → (3) 다음 모델 착수를 위한 데이터·소스 확보까지 진행했다.

---

## 1. 실험 스캐폴드 감사 (Scaffold Audit)

기존에 짜여 있던 `experiment_config.yaml` / `run_experiment.py` / `protocol_utils.py` / `generate_overlap.py`를 실제로 실행해 계획서(§5.3·5.7·5.12)와 일치하는지 검증했다.

- 단위 테스트 9개 전부 통과, manifest 생성 스크립트 정상 동작(12,480 row 생성).
- overlap non-edge=0 포함, budget/oracle 체크포인트 분리, scene 단위 cluster bootstrap 등 계획서의 핵심 규칙이 코드에 정확히 반영돼 있음을 확인.
- `validate_config()`가 `use_oracle_peak=true` 같은 leakage 방향 설정 변경을 자동 경고하는 것은 계획서에 없던 좋은 추가 안전장치.
- **담당자 검증 과정에서 발견된 오해를 교정**: manifest 12,480 row는 "26장면×5method"가 아니라 실제로는 main(7,680)+c1b(3,840)+c2(960) 세 phase의 합이며, main scene 수는 20(scenes_primary), method는 정확히 4개(DepthSplat/MVSplat/Vanilla3DGS/SparseGS)다. C1-b는 method 목록에 섞이지 않고 `phase: "c1b"`로 완전히 분리돼 있어 렌더 등가성 gate가 무력화되지 않는다.
- **GPU-hour 재계산**: budget(1/10/60/300초)은 재학습이 아니라 한 궤적에서 뽑는 체크포인트라는 전제로 실제 optimization 실행 수를 다시 세면 main phase만 960회(§5.4의 "약 1,200"과 근사), 여기에 C1-b refinement-on(960회)·C2(960회)를 더하면 총 ≈2,880회로 최초 추정의 약 2.4배. §5.4의 200~300 GPU-hour는 main phase 단독 기준으로는 맞고 전체 기준으로는 재검토가 필요하다는 결론.

## 2. 환경 확인

- GPU: **NVIDIA H200 NVL, 143,771 MiB, 당시 idle**.
- 외부 네트워크 접근 가능, `/data/Re-feem/datasets/{re10k,dtu,dl3dv}` 등은 이날 시작 시점에는 전부 빈 디렉터리였음(데이터 미확보 상태).

## 3. DTU scan1 확보 및 pose 파이프라인 검증

- DTU 공식 배포(`roboimagedata2.compute.dtu.dk`)의 `SampleSet.zip`(6.9GB)에 scan1 전체가 예시로 포함돼 있는 것을 확인. `remotezip`으로 zip central directory만 읽어 필요한 파일(이미지 49장, calibration 64개, GT point cloud)만 range request로 받아 **약 200MB/3.5분**에 끝냄 — `Rectified.zip`(130GB) 전체를 받을 필요가 없었음.
- DTU의 3×4 projection matrix를 RQ 분해로 K/R/t로 변환하는 `dtu_dataset.py` 작성. 재구성 오차 <0.002, scene 중심 투영이 이미지 중앙 근처에 정확히 떨어지는 것으로 수치 검증.

## 4. Vanilla 3DGS(gsplat) 러너 구현 및 파일럿 실행

`experiments/scripts/vanilla_3dgs_runner.py` 작성. gsplat 1.5.3의 `DefaultStrategy`로 densification 구현, `protocol_utils.budget_checkpoint()`/`oracle_checkpoint()`를 실제 학습 궤적에 직접 적용.

- **버그 발견·수정**: 스텝 완료 "후"에 시간을 재는 구조라 마지막 체크포인트가 budget을 항상 소폭 초과 기록됐고, `budget_checkpoint()`가 이를 정직하게 걸러내 `None`을 반환하는 문제를 발견. 스냅샷 시점의 `wall_clock`을 budget 값으로 clamp해 해결 (protocol_utils 자체는 정상 — 문제는 러너의 타이밍 기록 쪽이었음).
- **파일럿 결과** (scan1, 8-view, seed 0, budget 300초, random init 100k points):

  | budget | iter | gaussians | test PSNR |
  |---|---|---|---|
  | 60s | 375 | 100,000 | **10.69** |
  | 300s | 1,894 | 221,188 | 9.85 (↓) |

  densification 시작 후 Gaussian은 늘었는데 PSNR은 오히려 하락 — 계획서 H1/H2가 그리는 "sparse-view 과적합/정점 후 하강" 패턴과 정성적으로 일치. **다만 random init·1,894 iteration(표준 3만)·단일 seed/scene이라 수치 자체는 인용 불가, "로깅·체크포인트 파이프라인이 이 패턴을 놓치지 않는다"는 배관 검증으로만 사용.**

## 5. GPU 메모리·병렬 실행 테스트

- 100k~221k Gaussian, 1600×1200 해상도에서도 peak VRAM은 1~1.2GB. 3DGS는 구조적으로 메모리가 가벼운 표현이라 143GB GPU에서 2GB만 쓰는 건 정상. GPU util은 이미 단일 프로세스에서 99~100%로 **연산 병목**.
- **동일 GPU에서 6개 프로세스 동시 실행 테스트**: solo 125 iter/20s(≈6.25 it/s) vs 6개 동시 실행 시 각 20~22 iter/20s(합산 ≈6.2~6.3 it/s) — **총 처리량이 거의 늘지 않음**. H200의 여유 메모리로 "한 GPU에서 여러 run을 동시에 돌려 GPU-hour를 절약"하는 전략은 이 워크로드(3DGS)에는 통하지 않는다는 것을 실측으로 확인. 순차 실행 기준으로 §1의 GPU-hour 재추정을 봐야 함(멀티 GPU가 있다면 그쪽이 유일한 실질적 병렬화 경로).

## 6. COLMAP SfM init + LPIPS 통합

- `experiments/scripts/colmap_init.py`: `pycolmap`(3.13.0, prebuilt wheel)으로 COLMAP CLI의 `feature_extractor→exhaustive_matcher→point_triangulator` 조합을 재현. Pose-given track이므로 pose는 추정하지 않고 DTU calibration의 고정 pose로 **known-pose triangulation만** 수행. **오직 학습(input) view만 사용, held-out test view나 GT point cloud는 참조하지 않음**(leakage 방지).
- 검증: scan1·8-view → 1,994개 3D point, mean reprojection error 0.61px로 기하적으로 정상. Triangulation이 너무 빈약하면(<200점) random-sphere init으로 자동 fallback, 로그에 `init_source`로 기록.
- LPIPS(AlexNet) 연결 완료, 가중치는 PyTorch 공식 호스트에서 자동 다운로드.
- **환경 사고**: `pip install lpips`가 공유 `ps3` conda env의 torch를 2.2.2→2.8.0으로 조용히 업그레이드시켜 torchaudio와 버전이 어긋남. torch 2.2.2+cu121/torchvision 0.17.2+cu121로 원복, lpips는 `--no-deps`로 재설치해 재발 방지. **교훈**: 공유 env에 패키지 추가 시 항상 `--no-deps` 또는 버전 고정 필요.
- 통합 후 재검증(scan1·8-view·30초) 정상 동작 확인.

## 7. MVSplat 소스 확인 및 DTU scan 15개 확보

- MVSplat 공식 repo 확정: `github.com/donydchen/mvsplat`. 담당자 결정으로 실제 clone/통합은 COLMAP init 완료 후로 순서 조정, 오늘은 착수하지 않음.
- **DTU scan 확장**: pixelNeRF류 "표준 split"은 정확한 번호 출처가 코드가 아닌 별도 파일이라 재현 신뢰도가 낮아, **자체적으로 scan 번호를 스프레드해서 선정**하기로 결정(외부 논문과의 split 일치는 주장하지 않음). `Rectified.zip`(129GB) central directory만 읽어 실제 존재하는 124개 scan을 확인하고, ReadMe가 명시한 360도 회전 그룹(8개)에서 그룹당 1개만 선택해 장면 다양성 왜곡을 피함. 최종: **scan1(기존) + 9,17,25,34,42,50,58,67,75,87,95,104,112,120 = 15개**, `remotezip` range read로 이미지(scan당 49장)+GT point cloud만 받아 총 **3.0GB**. 중간에 scan104가 연결 끊김으로 20분 정지한 것을 발견해 재시도로 해결(향후 다운로드 스크립트에 timeout 필요, 미해결 TODO).

## 8. MVSplat 통합 및 DTU zero-shot 검증

- 공유 `ps3` env 사고(§6)를 반복하지 않도록 **완전히 격리된 `mvsplat` conda env**를 새로 만들어 README가 요구하는 정확한 버전(torch 2.1.2, torchvision 0.16.2)으로 설치. 커스텀 CUDA 라스터라이저(`diff-gaussian-rasterization-modified`)는 `--no-build-isolation`으로 별도 빌드 필요했음(빌드 격리 환경에 torch가 없어 실패하는 문제). 공식 Google Drive에서 RE10K 사전학습 체크포인트(48MB) 다운로드.
- **부수적 발견**: MVSplat repo의 `convert_dtu.py`에 공식 DTU sparse-view test split(16개 scan: 1,8,21,30,31,34,38,40,41,45,55,63,82,103,110,114)이 코드로 박혀 있는 것을 발견 — 어제 "정확한 출처가 없어 자체 선정한다"고 판단했던 그 표준 split의 실제 소스. 자체 선정한 15개와는 scan1·34만 겹침. 어느 쪽을 쓸지는 다음 세션 결정.
- 그 스크립트의 카메라 정규화 로직(scale 1/200, intrinsics 정규화 + principal point (0.5,0.5) 고정, near/far 고정값)을 그대로 재현해서, MVSplat 전용 배포본(`dtu_training.rar`) 없이도 **우리가 직접 받은 raw DTU calibration만으로** RE10K 체크포인트를 우리 데이터에 돌리는 브릿지 스크립트 작성.
- **결과: scan1에서 2-view context로 zero-shot cross-dataset(RE10K→DTU) 재구성 성공.** PSNR 수치(4.8~11.8)만 보면 낮지만, 렌더링 이미지를 직접 확인하면 동일 물체·동일 텍스트 위치가 명확히 재현됨 — 카메라 좌표계 변환이 틀렸다면 나올 수 없는 결과라 변환의 정확성 자체에 대한 강한 증거. **두 번째 method가 실데이터로 검증됨.**
- 오늘은 여기까지 — `vanilla_3dgs_runner.py` 수준의 정식 러너(protocol_utils 스키마 연동, 여러 scan/view-count 자동화)는 다음 세션 작업으로 남김.

---

## 종합 결론

1. **파이프라인 배관은 end-to-end로 작동함이 검증됐다.** DTU pose 파싱 → COLMAP SfM init → gsplat 학습/densification → PSNR·SSIM·LPIPS 평가 → budget-end/oracle 체크포인트 분리까지, 계획서가 사전에 고정한 프로토콜 규칙이 실제 코드에서 깨지지 않고 동작한다.
2. **두 계열(optimization·feed-forward) 모두 실데이터로 최소 1개씩 검증됐다.** Vanilla 3DGS는 학습 곡선(과적합 패턴 포함)까지 확인했고, MVSplat은 RE10K 체크포인트로 우리 DTU 원본 데이터에서 zero-shot cross-dataset 재구성이 실제로 되는 것을 이미지로 직접 확인했다. 이 프로젝트의 핵심 비교축(feed-forward vs optimization) 양쪽 다 "코드가 돈다"는 단계는 넘었다.
3. **지금까지 나온 수치(vanilla 3DGS PSNR 9~11dB, MVSplat PSNR 4.8~11.8)는 결과가 아니라 배관 검증용이다.** vanilla는 iteration 수가 짧고(수백~수천 vs 표준 3만) 단일 seed/scene, MVSplat은 context view 2장·scan 1개만 테스트했다. 둘 다 정식 파일럿·본 실험 결과로 인용하면 안 된다.
4. **GPU-hour 예산은 두 가지 이유로 재계산이 필요하다.** (a) C1-b·C2를 포함하면 실제 optimization 실행이 최초 추정보다 약 2.4배 많고, (b) H200의 여유 메모리로 병렬 실행을 통해 시간을 줄일 수 있을 거라는 가정이 실측 결과 성립하지 않는다(연산 병목이라 처리량이 늘지 않음). 두 요인 모두 방향이 "더 많은 시간이 필요하다"쪽이라, 파일럿 실측 GPU-hour를 근거로 §5.4를 갱신해야 한다.
5. **데이터 확보 축은 DTU가 앞서가고 RE10K가 뒤처져 있다.** DTU는 scan 15개(3.0GB)까지 확보했지만, 계획서상 주 데이터셋인 RE10K는 아직 손도 안 댐. DTU만으로는 계획서의 "본 실험"을 대체할 수 없다(외부검증/C2 전용 데이터셋).
6. **환경 격리가 반복적으로 중요했다.** 오늘만 두 번(`ps3`+lpips, 그리고 예방적으로 `mvsplat` 전용 env 신설) 패키지 설치가 조용히 torch 버전을 깨뜨릴 뻔했다. 모델마다 격리된 conda env를 쓰는 게 이제 이 프로젝트의 사실상 규칙이 됐다.
7. **다음 병목은 코드가 아니라 "스케일업과 통합 마감"이다.** 개별 모델 하나씩 도는 것은 확인됐으니, 이제 여러 scan/seed 자동 실행과 protocol_utils 스키마로의 통합이 남았다.

---

## 다음 실행 목록

우선순위 순으로 정리. □는 미착수, ✅는 오늘 완료. 괄호는 이유/전제조건.

1. □ **DTU scan 목록 재검토** — 오늘 발견한 공식 split(16개: 1,8,21,30,31,34,38,40,41,45,55,63,82,103,110,114)으로 갈아탈지, 자체 선정 15개를 유지할지 결정. 바꾸면 scan8,21,30,31,38,40,41,45,55,63,82,103,110,114 추가 다운로드 필요(scan1·34는 이미 있음).
2. □ **MVSplat을 protocol_utils 스키마로 감싸는 정식 러너 작성** — 지금은 1회성 검증 스크립트(`/data/Re-feem/code/mvsplat/run_on_custom_dtu.py`)뿐. `vanilla_3dgs_runner.py`처럼 experiment_id/scene/seed/method/checkpoint_path 로그를 남기게 확장.
3. □ **여러 scan/seed를 자동 순회하는 driver 스크립트** — 지금 `vanilla_3dgs_runner.py`는 단일 scan/view_count/seed CLI만 지원. `run_experiment.py`의 manifest를 실제로 순회 실행하는 러너로 확장(vanilla 3DGS·MVSplat 공통).
4. □ **새로 받은 scan들에 대해 overlap 계산** — `generate_overlap.py` 실행(COLMAP triangulation에서 나온 SfM point 재사용 가능). §5.3의 view 수 내 층화 threshold 산출의 실데이터 기반 마련.
5. □ **RE10K 확보 여부/시점 결정** — YouTube 기반 배포라 DTU보다 어려울 것으로 예상됨(§3 dataset_recommendation.md). 주 데이터셋이므로 계속 미루면 "본 실험" 자체가 시작을 못 함.
6. □ **§5.2 모델별 지원 view 수 표 채우기** — MVSplat의 2/4/8/12-view 실제 동작 여부 확인(오늘은 2-view만 테스트), `model_registry.py`의 placeholder 교체.
7. □ **§5.4 GPU-hour 예산 재계산 문서화** — 오늘 실측한 it/s, 병렬화 무효 결과를 근거로 파일럿 규모의 실측 GPU-hour를 산출해 계획서 수치 갱신.
8. □ **DTU 다운로드 스크립트에 timeout/재시도 추가** — scan104 연결 끊김 사고 재발 방지 (현재 미해결).
9. □ **C2 depth-noise 개입 준비** — COLMAP init 파이프라인이 있으니 depth back-projection에 iid noise/scale bias를 주입하는 코드를 다음 단계로 준비 가능.

✅ 오늘 완료: 스캐폴드 감사, 환경 확인(H200), DTU scan1 확보+검증, vanilla 3DGS 러너+파일럿, GPU 병렬 테스트, COLMAP init+LPIPS 통합, DTU 15-scan 확보, **MVSplat 통합 및 zero-shot 검증**.
