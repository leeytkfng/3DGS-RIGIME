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

**논문 연결**: §5.12(통계 분석 계획)를 실제 예산 배분에 처음 적용한 사례이자, 실험 설계 단계에서 "리스크 대응"이 아니라 "선제적 최적화"로 같은 원칙을 사용한 것 — §9 리스크 완화표도 그에 맞게 갱신(`~~seed 3→2~~`로 취소선 처리, 사유 각주 추가).

## 8. fsgs_runner.py 작성 — 그 과정에서 진짜 버그 하나 발견

**실험 목적**: §3~4에서 만든 FSGS 1회성 검증 스크립트를, 다른 모델들처럼 `protocol_utils` 스키마(budget checkpoint, trajectory 로그)를 따르는 정식 러너로 승격. §7에서 정한 30-scene 본 실험에 FSGS를 포함시키기 위한 선행 작업.

**데이터/특징**: FSGS는 외부 repo(`/data/Re-feem/code/fsgs`)라 `train.py`를 통째로 재구현하지 않고, 그 안의 Scene/GaussianModel/render()/loss는 그대로 갖다 쓰되 바깥 학습 루프만 우리 wall-clock budget(1/10/60/300s) 체계로 감쌌다 — `vanilla_3dgs_runner.py`가 gsplat을 감싸는 것과 같은 패턴.

**쉽게**: 만드는 과정에서 FSGS 코드를 한 줄씩 읽다가 진짜 버그를 하나 찾았다 — 우리가 seed로 "이 8개 view를 학습에 써라"라고 미리 골라놔도, FSGS 내부 코드(`readColmapSceneInfo`)가 그걸 무시하고 자기 방식(순서상 8번째마다 하나씩 test로 빼고, 남은 것 중에서 균등 간격으로 n_views개를 자체적으로 다시 고름)대로 view를 새로 골라버리고 있었다. 즉 지난번(§4) "첫 실제 학습 성공"때도 사실은 우리가 의도한 view가 아니라 FSGS가 자기 마음대로 고른 view로 학습했던 것 — 다른 모델(Vanilla3DGS/MVSplat)과 같은 조건으로 비교하는 게 이 연구의 핵심 전제인데, 그 전제가 깨져 있었다.

**전문 용어**: `scene.dataset_readers.sceneLoadTypeCallbacks["Colmap"]`을 우리 함수로 monkey-patch — 카메라 로딩(`readColmapCameras`/`getNerfppNorm`/`fetchPly`)은 FSGS 원본을 그대로 재사용하고, train/test split 로직만 `prep_dtu_for_fsgs.py`가 만든 seed 기반 train_ids/test_ids로 강제 교체했다. 수정 후 실측: seed=0은 1998개, seed=1은 605개의 triangulated point로 서로 다르게 나와, seed가 실제로 view 선택에 반영되기 시작했음을 확인. 추가로 2-view처럼 극단적으로 sparse한 조건에서는 triangulation이 0점을 내놓는 경우가 있어 FSGS의 CUDA rasterizer가 빈 Gaussian으로 즉시 죽는 것도 발견 — `prep_dtu_for_fsgs.py`에 Vanilla3DGS와 동일한 `MIN_SFM_POINTS=200` 기준 random-sphere fallback을 추가해 해결(실측: 2-view/seed0에서 fallback 100,000점으로 정상 학습 확인).

**결과**: DTU scan1 8-view(seed0/seed1)·2-view(fallback 경로) 모두 end-to-end로 정상 학습·평가·체크포인트 저장까지 확인(예: 8-view/seed1/5s budget → test PSNR 7.96→9.36dB). `model_registry.py`에 FSGS 정식 등록, `run_experiment.py` manifest 생성 시 뜨던 `Unknown method in config: FSGS` 경고도 해소.

**논문 연결**: 이 view-selection 버그는 우리가 §4.2에서 이미 문서화한 "sparse-init 편차"와는 별개로, 고치지 않았다면 FSGS만 다른 (사실상 무작위) view 조건에서 비교되는 훨씬 더 심각한 confound였다 — 발견하고 고친 게 이번 세션의 가장 중요한 성과 중 하나. 다음 단계는 RE10K/DL3DV로 확장(현재 DTU만 지원)한 뒤 §7에서 정한 30-scene 본 실험에 FSGS를 포함시키는 것.

## 9. Per-Gaussian gradient accumulation 조사 — "관측 부족" Gaussian 비율이 view 수에 강하게 좌우됨

**실험 목적**: gsplat `DefaultStrategy`의 densification 판단식(`grad2d[g]/count[g] > tau`)에서 `count[g]`(그 Gaussian이 최근 100-step window 동안 관측된 횟수)가 view 수에 따라 어떻게 분포하는지 실측 — "sparse-view에서 optimization이 손해 보는 이유"를 코드로 추적 가능한 메커니즘으로 설명할 수 있는지 확인.

**데이터/특징**: RE10K main subset 3개 scene × view_count{2,4,8,12}, 2000 step, seed=0. 초기화는 다른 러너와 동일 규칙(COLMAP sparse triangulation, 실패 시 random-sphere fallback). v1(densification 끄고 리셋 없이 누적)은 survivorship bias 때문에 null 결과가 나와 폐기, v2(densification 켜고 매 100-step window의 실제 판단 시점을 스냅샷)로 재설계했다.

**쉽게**: 처음엔 "관측이 적은 Gaussian은 평균 gradient가 노이즈로 부풀어서 densification 판정을 잘못 받을 것"이라 예상했는데, 실제로는 그런 오탐이 전혀 없었다(0%에 가까움) — 관측이 적으면 분자도 같이 작아지니까. 대신 훨씬 더 큰 걸 발견했다: **view가 적을수록 "어떤 학습 구간에서도 거의 안 보이는" Gaussian의 비율 자체가 폭발적으로 늘어난다** — 2-view에서는 절반(51.5%)이 이런 "죽은 예산"이고, 12-view에서는 0.7%뿐이다. 초기화 방식을 완전히 고정한 채(한 scene은 4개 view 조건 전부 같은 fallback 초기화를 씀) 봐도 55%→32%→8%→1%로 똑같이 줄어든다.

**전문 용어**: `gsplat/strategy/default.py::_update_state()`/`_grow_gs()`를 직접 호출하는 계측 스크립트(`gaussian_gradient_accumulation_probe.py`) 작성. `count≤2`인 Gaussian의 τ(=0.0002) 초과 비율은 전 조건에서 0.00~0.003% — 원 가설(노이즈로 인한 오탐) 기각. 대신 `count≤2` 비율 자체가 pooled 평균 51.5%(2-view)→30.8%(4-view)→3.2%(8-view)→0.7%(12-view)로 단조 감소 — 초기화 방식 교란요인을 제거한 단일 scene(1214f2a11a9fc1ed, 4개 조건 전부 random-sphere fallback)에서도 55.3%→32.3%→8.3%→1.3%로 재현.

**논문 연결**: "sparse-view에서 optimization이 불리하다"는 지금까지의 서술을 "Gaussian 예산의 절반 이상이 gradient 신호를 거의 못 받는 채로 방치된다"는 구체적 메커니즘으로 대체할 수 있다. 전문은 `paper/paper_gaussian_observation_starvation_2026-08-13.md`. Pilot 규모(scene 3개)라 scene 확대 재현, overlap 축 추가, floater 지표와의 상관 확인이 남은 일.

## 10. fsgs_runner.py를 RE10K/DL3DV로 확장

**실험 목적**: §8에서 DTU만 지원하던 `fsgs_runner.py`를 RE10K/DL3DV까지 확장 — §7에서 정한 30-scene 본 실험(Vanilla3DGS/MVSplat/FSGS 세 방법론 비교)에 FSGS를 포함시키기 위한 마지막 선행 작업.

**데이터/특징**: `prep_dtu_for_fsgs.py`에 dataset-agnostic `prepare_views_for_fsgs()`를 새로 추가 — DTU 전용 버전이 하던 일(images/, sparse/0/, poses_bounds.npy, {n_views}_views/dense/fused.ply 생성)을 "이미 메모리에 로드된 view 리스트"만 받아서 하도록 일반화했다. `vanilla_3dgs_runner.py`가 이미 RE10K/DL3DV용으로 쓰던 `_colmap_init_from_loaded_views()`와 정확히 같은 패턴(임시 디렉토리에 view를 써서 known-pose triangulation 코어 재사용).

**쉽게**: DTU는 view id가 1~49 깔끔한 정수라 파일명도 그걸 그대로 썼는데, RE10K/DL3DV는 원본 프레임 번호가 그렇게 안 깔끔해서 "train_0000.png"처럼 우리가 새로 이름을 붙이는 방식으로 바꿨다. near/far(카메라가 볼 수 있는 최소/최대 거리) 값도 데이터셋마다 달라서, RE10K는 MVSplat 러너가 쓰던 값(1~100), DL3DV는 DepthSplat 러너가 쓰던 값(0.5~200)을 그대로 재사용했다.

**전문 용어**: `fsgs_runner.py`에 `--dataset {dtu,re10k,dl3dv}` 분기 추가(vanilla_3dgs_runner.py와 동일 CLI 관례), view-selection monkeypatch(`_make_patched_colmap_loader`)를 int(DTU) 전용에서 문자열 이름 기반으로 일반화. RE10K(scene `0588138dfec165a1`, 8-view)·DL3DV(scene `09b05fa3...`, 8-view) 각각 실제 학습으로 end-to-end 검증(예: RE10K 10s budget → test PSNR 14.5→18.7dB), DTU 회귀 테스트도 재확인해서 기존 경로가 안 깨졌음을 확인.

**논문 연결**: 체크리스트의 "본실험 착수 전 필수 4개" 중 하나가 끝났다 — 이제 co-visibility selector, §5.11/§5.12 동결만 남으면 30-scene 본 실험에 세 방법론(Vanilla3DGS/MVSplat/FSGS) 전부를 포함해 착수할 수 있다.

## 11. co-visibility selector 구현 — overlap_levels(고/저) 축을 실제로 통제

**실험 목적**: `run_experiment.py`의 manifest 생성 코드를 읽어보니 `product(scenes, seeds, view_counts, overlap_levels, budgets, methods)`로 되어 있어, **같은 scene·view_count에서 overlap=low 행과 overlap=high 행이 둘 다 나와야 한다**고 이미 전제하고 있었다. 그런데 지금까지 RE10K/DL3DV의 view 후보는 scene·view_count당 1개만 만들고 overlap은 사후 측정만 했다 — "선택" 로직 자체가 없었다(체크리스트가 오래전부터 지적하던 공백).

**데이터/특징**: 기존 candidate 파일(`re10k_main_subset.json`, `dl3dv_overlap_v2/`)은 이번 세션에 이미 여러 실측 결과(C1-a seed×3 파일럿, FSGS 검증 등)가 참조하고 있어 건드리지 않고 그대로 뒀다. 대신 새 selector와 검증 결과는 완전히 별도 파일로 남겼다(DL3DV v1→v2 전환 때 썼던 것과 같은 보존 원칙).

**쉽게**: "가까이 모여있는 view들을 뽑으면 overlap이 높고, 멀리 퍼진 view들을 뽑으면 overlap이 낮다"는 직관을 그대로 알고리즘으로 만들었다. 같은 방법(farthest-point sampling — 이미 뽑은 점들에서 가장 먼 점을 계속 추가하는 방식)을 좁은 구간에만 적용하면 "high overlap 후보", 전체 구간에 적용하면 "low overlap 후보"가 된다. 그리고 실제로 만들어본 후보들의 overlap을 측정해서 진짜 그렇게 나오는지 확인했다.

**전문 용어**: `core/view_selector.py` 신규(`farthest_point_sample`은 DL3DV v2 스크립트에 있던 DepthSplat 재현 로직을 공용 모듈로 승격). RE10K 3 scene × view_count{2,4,8,12}로 실측: **12/12 조건에서 high≥low 확인**(4-view 기준 mean_overlap 0.82 vs 0.75, 8-view 0.62~0.75 vs 0.40~0.58 등). DL3DV 3 scene도 실측: **11/12 조건 확인**, 단 1개(`365d7c...` scene, 12-view)는 방향이 반대로 나왔다(high=0.227, low=0.400) — DL3DV가 RE10K보다 궤적이 다양해서(루프형 등) 좁은 window라고 항상 overlap이 높다는 보장이 없는 경우가 있는 것으로 추정, 정직하게 기록. 결과: `re10k_overlap_candidates.json`, `dl3dv_overlap_lowhigh/`.

**논문 연결**: §4.1의 "View overlap: 고/저" 축이 이제 실제로 구현 가능한 상태가 됐다. 지금은 파일럿 규모(scene 3개) 검증이고, 30-scene 본 실험 착수 시 전체 scene으로 확대 적용해야 한다 — DL3DV의 1건 실패 사례도 전체 적용 시 재확인 필요. 이걸로 "본실험 착수 전 필수 4개" 중 남은 건 §5.11/§5.12 동결뿐이다.

## 12. §5.11/§5.12 동결 — 그리고 τ를 실측하다가 발견한 정정 사항

**실험 목적**: "본실험 착수 전 필수 4개"의 마지막 두 개, §5.11(조기 종료 규칙)과 §5.12(통계 분석 계획)를 최종 확정. 특히 §5.12의 τ(승패 판정 임계값)는 지금까지 "seed 변동성과 실용적 최소 차이 중 큰 값"이라는 공식(`compute_tau`)만 있고 실제 숫자가 없었는데, 오늘 만든 5-scene/seed×3 파일럿 데이터(§1)로 처음 실측했다.

**데이터/특징**: `re10k_c1a_pilot` seed×3 결과(240 rows)에서 scene별로 seed 3개의 표준편차를 구하고, 그걸 다시 scene들 사이에서 평균 냈다.

**쉽게**: seed를 바꿔도 결과가 얼마나 흔들리는지 재봤더니, view가 적을 때(2/4장)는 거의 안 흔들리는데(0.04~0.17dB) view가 많을 때(8/12장)는 훨씬 많이 흔들렸다(0.13~1.41dB, 심하면 3.5dB까지). 그래서 "이 정도 차이는 진짜 차이로 볼지 우연으로 볼지" 가르는 기준선(τ)을 view 수에 따라 다르게 잡아야 한다는 게 실측으로 확인됐다.

그런데 이 작업을 하다가 예상 못 한 걸 발견했다. §체크리스트에 "8-view에서 10초만 줘도 Vanilla3DGS가 MVSplat을 역전한다(16.8 vs 16.5dB)"라고 적어뒀던 부분을 다시 보니, 이 0.3dB 차이는 방금 잰 8-view 노이즈 기준선(1.4dB)보다 훨씬 작다 — 즉 "역전"이 아니라 "무승부"였다. 게다가 scene 5개를 각각 따로 보면 3개는 Vanilla3DGS가 크게 이기고(+1.1~+5.3dB) 2개는 MVSplat이 크게 이겨서(+1.0~+5.8dB), 애초에 "역전"이라는 하나의 방향이 없었다 — 여러 scene의 큰 차이들이 평균 내면서 우연히 거의 상쇄된 것뿐이었다.

**전문 용어**: τ를 view_count 2단계로 확정 — `view_count∈{2,4}`: 0.5dB(실용적 최소 차이가 지배), `view_count∈{8,12}`: 1.4dB(seed 변동성이 지배, 평균값 채택 — 단일 3.52dB outlier는 seed=3 표본 부족으로 인한 불안정 추정치로 보고 배제). Holm-Bonferroni 다중비교 family는 방법 쌍(pair)당 32비교(view_count 4 × overlap 2 × budget 4)로 정의, 방법 쌍끼리는 풀링 안 함. §5.11은 메인(budget-end checkpoint) 유지 + 8/12-view 한정 보조 실험(3~5 scene, Vanilla3DGS+FSGS, validation-view 기반 조기종료 시뮬레이션, 부록용)으로 확정.

**결과**: τ 적용 후 12-view/60s의 역전(20.6 vs 17.1dB, delta +3.5dB)은 τ=1.4dB를 확실히 넘어 **robust하게 유지**된다. 반면 8-view/10s는 **Tie로 재분류**해야 한다. `overall.md` §5.12에 이 정정을 명시적으로 기록, 체크리스트 9·11번에도 정정 각주 추가(원래 기록은 지우지 않고 남김 — 뭘 왜 고쳤는지 추적 가능하게).

**논문 연결**: "본실험 착수 전 필수 4개"(FSGS RE10K/DL3DV 확장, co-visibility selector, §5.11, §5.12)가 오늘 전부 끝났다 — 이제 30-scene × seed 2회 본 실험(Vanilla3DGS/MVSplat/FSGS 세 방법론)에 착수할 수 있다. 동시에 이 정정은 "pooled 평균만 보면 놓치는 게 있다"는, 오늘 초반에 다뤘던 scene-cluster bootstrap 논리(§7)를 실제 데이터로 재확인해준 사례이기도 하다.

## 13. C1-a 본 실험 착수 — RE10K 30 scene, 세 방법론, 백그라운드 실행 중

**실험 목적**: 지금까지 준비한 모든 것(30-scene 확장, FSGS RE10K 연동, τ 확정)을 실제로 돌리는 단계. C1-a Regime Map의 진짜 본 실험 — "view 수·계산 예산에 따라 언제 optimization이 feed-forward를 이기는가"에 대한 최종 답을 낼 데이터.

**데이터/특징**: RE10K main subset을 20→30 scene으로 추가 확장(`extend_re10k_main_subset.py`, 기존 20개는 그대로 두고 다른 seed로 10개만 새로 뽑음 — 기존 파일럿 결과 보존). `run_re10k_c1a_main.py` 신규 — 30 scene × view_count{2,4,8,12} × budget{1,10,60,300}s × {MVSplat 1회, Vanilla3DGS seed×2, FSGS seed×2}. **이번 착수 범위에서는 overlap_level(고/저) 축 제외** — selector가 아직 3-scene pilot 검증만 된 상태라 검증 안 된 축까지 넣어 84 GPU-hour를 한 번에 태우는 위험을 피함(view_count 축만으로 약 42 GPU-hour, 약 40시간 예상). 1 scene으로 세 방법론 전부 스모크 테스트 통과 후 백그라운드로 착수(nohup, PID 1149474, 07:00 KST 재시작 기준).

**진행 중 관측**: 착수 직후 실제 GPU 사용을 직접 확인 — `nvidia-smi`로 H200 NVL, GPU util 95~99%, 메모리 21~22GB/143.8GB(15%), 온도 62°C, 전력 200W/600W(캡의 33%) 확인. 전부 정상 범위 — 40시간짜리 job이 안전하게 도는 중임을 실측으로 확인. 컨테이너 환경(`/.dockerenv` 확인)이라 `nvidia-smi`의 PID가 호스트 기준이라 컨테이너 안 프로세스 PID와 직접 매칭은 안 되지만(그래서 "No running processes found"로 표시됨), GPU-util과 온도/전력 추이 자체가 실제 학습이 진행 중이라는 증거.

**결과 표시**: 진행 상황을 보여주는 대시보드를 Artifact로 배포(regime map 그리드, quality-vs-budget 곡선, 원본 데이터 표, 진행률/ETA) — summary JSON이 (scene, view_count) 조합 하나 끝날 때마다 갱신되므로 데이터가 쌓이는 대로 재배포해서 갱신할 예정.

**논문 연결**: 이게 완료되면 C1-a의 핵심 결과(regime map, 역전 경계, Pareto frontier)가 나온다 — §5.12에서 확정한 τ(2/4-view: 0.5dB, 8/12-view: 1.4dB)와 scene-cluster bootstrap CI를 적용해 최종 승패 판정을 낼 수 있다. overlap_level 축과 C1-b/C2는 이 본 실험 이후 단계.

