# 일일 보고서 — 2026-08-13

## 오늘 목표

어제(8/12) 마무리한 C1-a 첫 파일럿(RE10K 5 scene, seed 1회)에서 나온 "역전 패턴"(view가 늘고 예산이 커질수록 Vanilla3DGS가 MVSplat을 역전)이 seed에 흔들리지 않는 진짜 패턴인지 확인하는 것부터 시작. 이어서 daily_logs/checklist에 이월된 나머지 항목들(renderer_equivalence_tolerance 동결, DepthSplat DL3DV v2 재실행, SparseGS 착수 등)을 처리한다.

---

## 1. C1-a 파일럿에 seed 3회 추가

**쉽게**: 어제 만든 첫 결과는 딱 한 번만 돌린 거라, "운이 좋아서/나빠서" 나온 우연한 숫자인지 아니면 진짜 패턴인지 아직 모른다. 그래서 똑같은 조건을 다른 랜덤 시드(=학습 순서를 섞는 난수 시작값)로 2번 더 돌려서, 매번 비슷한 결과가 나오는지 확인한다.

**전문 용어**: `run_re10k_c1a_pilot.py`에 `--seeds` 다중 지원 추가. MVSplat은 `deterministic=True`로 추론하므로 seed와 무관하게 결과가 항상 같아 1회만 실행하고, Vanilla3DGS만 seed(0/1/2)별로 반복한다(seed는 학습 순서 셔플에만 영향). 2-seed 스모크 테스트 통과 확인 후 RE10K 5 scene × 2/4/8/12-view × seed{0,1,2} 전체 실행 중(seed 0은 어제 결과 재사용, 1·2만 신규 실행 — 백그라운드, 완료되는 대로 아래에 추가 예정).

**논문 연결**: overall.md §5.12(통계 분석 계획)의 "seed는 scene 내 반복 측정으로 처리"라는 원칙을 처음으로 실제 데이터에 적용하는 단계. 이 결과가 있어야 어제 파일럿의 역전 패턴을 "우연이 아니다"라고 주장할 최소한의 근거가 생긴다.

## 2. renderer_equivalence_tolerance 최종 동결

**쉽게**: 어제 "이 정도 오차는 봐줘도 되는 기준선(33dB)"을 딱 1개 장면 데이터로 임시로 정했었다. 오늘은 그동안 여러 실험을 돌리면서 저절로 쌓인 데이터(130번의 검사, 390개의 개별 측정치)를 다 모아서 "그 기준이 진짜 맞는 기준이었나"를 다시 확인했다.

**전문 용어**: `experiments/outputs/`를 재귀 탐색해 모든 `renderer_equivalence_gate.json`을 모아 view별 PSNR 분포를 재계산. MVSplat/RE10K 240 샘플 + DepthSplat/DL3DV 150 샘플, 전체 median 45.21dB, min 26.76dB. PSNR≥33dB 기준으로 미달은 390개 중 9개(2.3%)뿐이고, 재구성 품질이 실제로 나오는 범위(8~25dB대)와는 충분히 떨어져 있어 혼동 위험이 낮다.

**논문 연결**: §5.8("허용 오차는 파일럿 전에 수치로 동결한다")의 마지막 미결 항목을 닫았다. 어제 이미 이 값(33dB)으로 C1-b 스케일업을 다 돌려놨기 때문에 소급 변경 없이 그대로 확정. `overall.md`/`experiment_config.yaml`에 반영.

## 3. SparseGS vs FSGS 결정 + FSGS 환경 구축

**쉽게**: 지금까지 우리 실험엔 "AI가 한 번에 만드는 모델"(MVSplat/DepthSplat)과 "일반 3D 최적화"(Vanilla 3DGS)만 있었는데, sparse-view 상황에 특화된 최적화 기법(3번째 방법론)이 아직 없었다. 후보 두 개(SparseGS, FSGS) 중 하나를 골라서 설치했다.

**전문 용어**: GitHub 메타데이터 비교(별/이슈/의존성) 후 **FSGS 선택** — SparseGS는 추가로 BoostingMonocularDepth 체크포인트를 수동 다운로드해야 하는 반면 FSGS는 MiDaS DPT-Hybrid를 `torch.hub`로 자동 처리하고, `--n_views` CLI가 우리 연구의 핵심 축(view 수)과 정확히 일치한다. 저장소를 `/data/Re-feem/code/fsgs`에 clone하고 `fsgs` conda env를 새로 만들었다. 공식 환경(CUDA 11.6/torch 1.12.1)은 H200(Hopper, compute capability 9.0)에는 너무 오래돼서 MVSplat 때와 같은 방식으로 torch 2.1.2+cu121로 교체했다. 빌드 중 `setuptools`가 최신(83.0.0)이라 `pkg_resources`가 빠져있어 커스텀 CUDA 확장(simple-knn, diff-gaussian-rasterization-confidence) 빌드가 실패하는 걸 발견 — `setuptools<70`으로 낮춰서 해결했고, `TORCH_CUDA_ARCH_LIST=9.0`을 명시해 둘 다 빌드 성공, import 검증과 `train.py --help` 완주까지 확인했다.

**논문 연결**: 지금까지 실험은 "일반 3DGS vs FF 모델" 비교였는데, "sparse-view 특화 3DGS vs FF 모델" 비교가 빠지면 "특화 기법을 안 써서 optimization이 불리했던 것 아니냐"는 반박이 가능하다. FSGS가 붙어야 C1-a Regime Map의 "optimization" 쪽 대표성이 완성된다. 아직 실제 학습은 안 돌려봤고 환경만 준비된 상태 — 다음은 데이터 포맷 맞추기와 protocol_utils 스키마 러너 작성.

## 4. FSGS 실제 학습 1회 성공

**실험 목적**: FSGS가 우리 데이터에서 실제로 돌아가는지 검증(아직 protocol 통합 전 단계). 환경만 만들어놨지 진짜 학습 코드가 우리 데이터 포맷을 소화하는지는 확인 안 했었다.

**실험 데이터/특징**: DTU scan1, seed=0, 12-view(다른 러너들과 같은 held-out 규칙: test는 1,8,...,43번 view). FSGS는 원래 `colmap patch_match_stereo`로 만든 dense point cloud를 요구하는데, 우리 시스템엔 COLMAP CLI(dense MVS 모듈)가 아예 없어서(pycolmap 파이썬 바인딩만 있음) 대신 우리가 이미 갖고 있는 sparse triangulation 결과(다른 러너와 동일 코어)를 그 자리에 채워 넣는 방식으로 우회했다(`prep_dtu_for_fsgs.py` 신규). Vanilla3DGS도 원래 sparse COLMAP init이라 방법론적으로 크게 벗어나지 않는다.

**돌리면서 발견한 이슈 2개** (쉽게 설명): (1) FSGS는 옛날 LLFF 방식의 관례를 그대로 물려받아서 "카메라가 얼마나 가깝고 먼 곳까지 보는지"(근/원 깊이 경계)를 담은 `poses_bounds.npy`라는 파일도 따로 요구했다 — MVSplat DTU 설정에서 쓰던 값(near=2.125, far=4.525)을 그대로 갖다 썼다. (2) 이미지 폴더 이름 기본값이 `images_8`(LLFF에서 원본 해상도를 8배 줄인 폴더를 쓰는 관례)이었는데 우리는 `images`로 만들어서 처음엔 빈 목록 에러가 났다 — 폴더명을 명시해서 해결.

**결과**: 1000 iteration(짧은 검증용, 정식 10000이 아님) 학습 완료. test PSNR 12.81dB / SSIM 0.562 / LPIPS 0.548 — 짧게 돌려서 낮지만, 학습이 끝까지 돌고 진짜 평가 숫자가 나왔다는 게 핵심.

**논문 연결**: 이걸로 세 번째 방법론(sparse-view 특화 optimization)이 우리 데이터에서 실제로 돌아간다는 게 확인됐다. 아직 protocol_utils 스키마를 따르는 정식 러너는 아니고, RE10K/DL3DV 확장도 안 했다 — 다음 단계.

## 5. C1-a seed 3회 결과 — 역전 패턴이 seed에 안정적

**실험 목적**: 어제 seed 1회로 나온 "8-view 이상에서 Vanilla3DGS가 MVSplat을 역전"이라는 패턴이 우연이 아닌지 확인(§1의 후속).

**실험 데이터**: RE10K 5 scene × 2/4/8/12-view × seed{0,1,2}, budget[1,10,60]s. MVSplat은 seed 무관(1회만), Vanilla3DGS만 3회 반복 — 240 rows.

**결과**: 2/4-view는 seed 간 표준편차 0.03~0.11dB로 거의 흔들림 없이 MVSplat이 압승. 8-view 이상에서 Vanilla3DGS가 역전하는 방향은 seed 3개 전부 일관됨(12-view/10s만 표준편차 1.13dB로 변동이 좀 있지만, 셋 다 MVSplat보다는 높거나 비슷).

**해석**: 파일럿 규모(5 scene)치고 역전 경계(4-view와 8-view 사이)가 꽤 안정적으로 재현됐다. 다만 seed 개수(3)와 scene 개수(5) 둘 다 아직 정식 신뢰구간(scene cluster bootstrap, `protocol_utils.py::scene_cluster_bootstrap_ci`)을 계산하기엔 작다 — 20 scene으로 확장해야 §5.12 정식 통계를 낼 수 있다.

## 6. 논문 수식 전체 정리 문서 작성

지금까지 코드/여러 문서에 흩어져 있던 수식(overlap, Gauss-Newton 공분산, 승패판정 τ, budget checkpoint 규칙, scene cluster bootstrap 신뢰구간, Holm 보정, C2 depth 모델, 3DGS 렌더링 핵심 수식)을 목적/사용처/해석과 함께 한 문서로 정리했다: `paper/paper_equations_reference.md`. 신뢰구간(§6)에 특히 상세히 — 왜 scene을 독립 단위로 봐야 하는지, run 개수로 세면 왜 과신하게 되는지까지 풀어썼다.

## 7. seed 3→2 / scene 20→30 전환 — 같은 예산에서 통계적 힘 키우기

**실험 목적**: §5의 결과가 다음 단계(main subset 20~30 scene 스케일업)로 갈 때 seed와 scene 중 어느 쪽을 늘리는 게 맞는지 결정. §6에서 정리한 신뢰구간(scene cluster bootstrap) 공식을 실제 grid 설계에 처음 적용하는 단계.

**데이터/근거**: (1) `overall.md` §9의 기존 리스크 완화표에 "GPU 시간 초과 시 축소 순서: seed 3→2가 1순위"가 이미 적혀 있었음을 재확인. (2) `scene_cluster_bootstrap_ci`(`protocol_utils.py`)는 scene을 리샘플 단위로 쓰므로 CI 폭이 seed가 아니라 scene 수(1/√scene)에 좌우된다는 원칙. (3) 실제 파일럿 로그(`/tmp/c1a_pilot_seeds_full.log`)에서 60초-budget trajectory 실측 소요시간(오버헤드 4.35s)을 뽑아 §5.4의 이론 추정(8.15s)과 대조.

**쉽게**: seed를 늘리는 건 "같은 문제를 다른 순서로 여러 번 푸는 것"이라 우연성만 줄여줄 뿐, scene을 늘리는 것처럼 "아예 다른 문제를 더 풀어보는 것"만큼 결론을 넓혀주지 못한다. 그런데 재밌게도 `20 scene × 3 seed`와 `30 scene × 2 seed`는 총 계산량(GPU-hour)이 완전히 똑같다(20×3=30×2=60) — 그러니까 **돈을 더 안 써도** scene을 10개 늘리고 seed를 1개 줄이기만 하면 신뢰구간이 더 좁아진다(공짜 개선).

**전문 용어**: trajectory 수 ∝ scene×seed×view×overlap×method이므로 scene×seed곱이 불변이면 총 GPU-hour(≈250h)도 불변. `run_experiment.py`로 manifest를 재생성해 main phase row 수가 7,680으로 변하지 않음을 실측 확인. CI 폭 축소율은 √(20/30)≈0.816 → 약 18.4%(20→40이면 29.3%). `experiment_config.yaml`(`seeds: [0,1]`, `scenes_primary: 30`)과 `overall.md` §5.4/§9에 반영. DTU는 GT geometry 외부 검증 세트일 뿐 통계 주 추론 단위가 아니므로(§4.2) scene 확대 대상에서 제외(8 scene 유지).

**FSGS 초기화 편차도 같이 정리**: FSGS 원 프로토콜은 dense-MVS 초기화를 쓰지만 우리는 COLMAP dense 모듈이 없어 Vanilla3DGS와 같은 sparse init을 공유시켰다(§3 참고). 이를 논문 methods/limitations 양쪽에 명시적으로 적어야 한다는 점, FSGS가 Vanilla3DGS를 못 이기더라도 그 자체로 유효한 결과(SplatFormer의 유사 사례)라는 점, dense-MVS 소규모(3~5 scene) 비교는 낮은 우선순위 백로그로 `overall.md` §4.2에 문서화했다.

**논문 연결**: §5.12(통계 분석 계획)를 실제 예산 배분에 처음 적용한 사례이자, 실험 설계 단계에서 "리스크 대응"이 아니라 "선제적 최적화"로 같은 원칙을 사용한 것 — §9 리스크 완화표도 그에 맞게 갱신(`~~seed 3→2~~`로 취소선 처리, 사유 각주 추가). 다음 우선순위는 `fsgs_runner.py`(protocol_utils 스키마) 작성과 `model_registry.py` FSGS 등록 — 현재 manifest 생성 시 `Unknown method in config: FSGS` 경고가 뜨는 상태라, 이게 끝나야 30-scene 본 실험에 FSGS를 포함할 수 있다.

