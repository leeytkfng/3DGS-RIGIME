# 실험 진행상황 체크리스트

이 파일은 살아있는 체크리스트다. daily_report_*.md는 그날그날의 기록(과거형)이고, 이 파일은 지금 시점의 상태(현재형)다. 항목을 완료하면 여기서 바로 `[ ]` -> `[x]`로 바꾸고, 근거(로그/파일 경로)가 없는 완료 표시는 하지 않는다.

마지막 검증: 2026-08-12, 실제 파일/로그/git 상태를 직접 확인해서 채움 (다른 세션이 자체 보고한 내용도 파일 존재 여부로 재검증함).

---

## 연구 설계 / 프로토콜

- [x] 연구 방향 문서화 (`experiments/direction/overall.md`)
- [x] 실험 config 작성 (`experiments/configs/experiment_config.yaml`)
- [x] budget checkpoint / oracle checkpoint 분리
- [x] scene 단위 cluster bootstrap 규칙
- [x] overlap non-edge=0 포함 규칙 구현
- [x] C1-b / C2 통제 실험 축 설계 (개념 설계만 — 구현은 "모델/러너" 섹션 참고, 별개임)
- [x] dense-view sanity 필요성 반영 및 실행
- [x] main benchmark 해상도 = checkpoint 상속 원칙 확정
- [x] §5.2 모델별 지원 view 수 표 완성 — 2026-08-12, MVSplat/DepthSplat 공식 config·README 기준으로 채움, `model_registry.py`의 `supports_views`도 동기화(MVSplat→`[2]`, DepthSplat→`[2,4]`).
- [x] §5.4 GPU-hour 예산 재계산 — 2026-08-12, manifest trajectory 수 × 실측 wall-clock 기반으로 재계산, 이후 250 GPU-hour로 확정(아래 C2 budget 결정 항목 참고).
- [ ] confidence 출력 열 — MVSplat/DepthSplat 둘 다 confidence/uncertainty 출력이 소스코드에 없음을 확인(grep 결과 0건). 표에는 "없음"으로 채워졌으나, 논문에서 이걸 어떻게 다룰지(대체 지표 필요한지) 결정 안 됨
- [x] C2 budget 결정 — 2026-08-12. main phase/C1-b와 동일하게 단일 300s trajectory + `budget_snapshots=[1,10,60,300]`로 확정(근거: overall.md §5.9). `experiment_config.yaml`/`run_experiment.py`에 반영, manifest 재생성으로 확인(c2 row에 `max_budget_seconds`/`budget_snapshots` 필드 생김). §5.4 GPU-hour 합계도 186~250h 범위에서 **250h로 확정**
- [ ] §5.11 조기 종료 규칙 동결 (미결 항목으로 문서에 이미 표시돼 있음)
- [ ] §5.12 통계 분석 계획 최종 동결

## 데이터

- [x] `/data/Re-feem` 디렉토리 구성
- [x] DTU 공식 split 16 scan + extra 13 scan = 29 scan
- [x] RE10K test 114 scene (probe 41 + mirror 73)
- [x] DL3DV pilot 25 scene, 공식 benchmark split과 중복 0 확인
- [x] RE10K/DL3DV `SOURCE.md` 작성 — 2026-08-12: RE10K SOURCE.md의 "train split 없음" 오기 정정. 실제로 `train/`에 39 scene 존재(8/10 `re10k_subset.zip` 다운로드에 원래 딸려 온 것, `critical_path_2026-08-10.md`에 이미 기록돼 있었음). 데이터 자체는 문제없고 의도적 미사용 상태 — 문서만 안 맞았던 것
- [x] RE10K main subset(20~30 scene) index 생성 — 2026-08-12, `generate_re10k_main_subset.py` 작성·실행. 로컬 114 scene ∩ MVSplat 공식 evaluation index(`assets/evaluation_index_re10k.json`, non-null 6,474건) ∩ frame수≥50 ∩ context/target 안 겹침 조건으로 96개 후보 중 seed=0으로 20개 결정론적 선정. 2-view는 공식 context/target 그대로, 4/8/12-view는 target(3-view, view_count 불문 고정)을 제외한 pool에서 seeded 생성. 출력: `experiments/outputs/re10k_main_subset/re10k_main_subset.json`. 검증: 20 scene × 4 view_count 전부 context/target 겹침 없음, index 범위 안(스크립트로 재확인). **부수 발견**: 공식 index 자체에 context/target이 겹치는 scene이 2개 있었음(`aadc1e2dc74fd644`, `cdf439b17a6a98d4`) — leakage 방지를 위해 main subset에서 제외
- [x] RE10K 2/4/8/12-view candidate에 대한 overlap 계산 — 2026-08-12. `colmap_init.py`를 DTU 전용에서 데이터셋 무관 공용 코어(`triangulate_sfm_points_from_cameras`)로 리팩터(DTU 경로는 얇은 wrapper로 남겨 기존 동작 그대로 유지 — scan1 4-view 313 SfM point로 재검증). RE10K 전용 로더 `core/re10k_dataset.py`(`.torch` chunk에서 필요한 frame만 디스크로 풀고 정규화된 카메라를 픽셀 K로 변환) + `analysis/generate_re10k_view_overlap.py` 신규 작성. main subset 20 scene × 4 view_count = 80 combo 전부 COLMAP 실행 완료(`experiments/outputs/re10k_main_subset/overlap/all_scenes_summary.json`). **핵심 발견**: 2-view는 20 scene 전부 mean_overlap=0.000(SfM 매칭 0건) — DTU에서 봤던 "2-view SfM 붕괴"가 RE10K에서도 100% 재현됨(MVSplat 공식 2-view context가 SfM 매칭 목적이 아니라 wide-baseline NVS 목적으로 뽑혀서 그런 것으로 추정). 4/8/12-view는 정상 범위(median 0.80/0.55/0.52)
- [x] RE10K/DL3DV용 overlap bucket (low/high threshold) 산출 — RE10K는 §5.3 stratify_thresholds_within_view_count 원칙대로 view_count별 median split: 4-view 0.804, 8-view 0.552, 12-view 0.524. 2-view는 분리 불가(전부 0). 결과: `experiments/outputs/re10k_main_subset/overlap/bucket_thresholds.json`.
  - **DL3DV 이식 완료** (2026-08-12, `core/dl3dv_dataset.py` + `analysis/generate_dl3dv_view_overlap.py` 신규): 카메라 규약은 DepthSplat 공식 변환 스크립트(`convert_dl3dv_train.py`)로 재현(blender c2w → opencv w2c). pilot 25 scene × 4 view_count = 100 combo 실행. 결과: 2-view는 25개 전부 overlap 0(RE10K/DTU와 동일 패턴, 세 번째 데이터셋 재현). **4-view도 14/25(56%)가 overlap 0** — RE10K(4-view 전부 정상)보다 훨씬 나쁨. 8-view부터 전부 정상(median 0.288), 12-view median 0.220.
  - **중요한 한계**: RE10K는 MVSplat 공식 context/target index(자기들 학습 시 쓰는 근접-프레임 제약이 이미 반영됨)를 그대로 썼지만, DL3DV는 그런 공식 index 파일이 없어서(`generate_dl3dv_index.py`는 scene_key→chunk 매핑만 만들 뿐, view 선택 index는 아님 — 확인함) 전체 250~400 프레임 중 순수 랜덤으로 뽑았다. 그래서 "4-view도 56%가 overlap 0"이라는 수치는 DL3DV 자체의 성질이라기보다 **우리 view 선택 방식이 DepthSplat 분포보다 더 넓게 퍼진 탓일 가능성이 큼**.
  - **DepthSplat 실제 test-time 샘플링 알고리즘 확인함**(`view_sampler_bounded_v2.py::sample()`, 2026-08-12 코드 리딩): test 시점엔 `context_gap = max_distance_between_context_views`로 고정(무작위 아님), `index_context_left = 0`으로 고정(즉 매 scene의 **첫 프레임부터 시작**), window `[0, context_gap]` 안에서 `extra_views_sampling_strategy="farthest_point"`(config 값)로 `num_context_views`개를 뽑는다. 우리 config 기준 4-view는 `context_gap=50`(=`max_distance_between_context_views`). 8/12-view용 gap 값은 별도 config가 없어 확인 못 함. 공정한 재실행을 하려면 이 알고리즘(고정 window + farthest-point-sample, 전체 영상 랜덤 아님)을 그대로 구현해야 함 — 다음 항목으로 기록, 아직 미착수
- [ ] RE10K citation/license 문구 논문용 정리

## 모델 / 러너

- [x] Vanilla3DGS runner (COLMAP init, LPIPS, densification 궤적 로깅) — `experiments/scripts/runners/vanilla_3dgs_runner.py`
- [x] MVSplat 정식 runner (protocol_utils 스키마) — `experiments/scripts/runners/mvsplat_runner.py`
- [x] DepthSplat probe (정식 runner 아님) — `experiments/scripts/probes/depthsplat_dl3dv_probe.py`
- [x] model_registry 구조 + batch driver 일반화
- [x] DTU 공식 split 16 scan x 2-view x seed0 batch
- [x] DTU scan1 2/4/8/12-view 통제 스모크 — Vanilla3DGS/MVSplat 둘 다 8개 run 전부 `status: ok` (`experiments/outputs_smoke_20260811/logs/batch_summary.json` 직접 확인)
- [x] DepthSplat probe -> 정식 runner 승격(부분) — 2026-08-12 밤, `runners/depthsplat_dl3dv_runner.py`가 wall_clock/test_psnr/gaussian_count를 protocol 스키마로 로깅하며 C1-b 파이프라인에 연결됨(§V3 참고). 다만 아직 DL3DV 전용(RE10K 체크포인트 미다운로드)이고 budget_checkpoint 등 full trajectory 로깅은 없음 — 완전한 "정식 runner"까지는 아님
- [x] SparseGS/FSGS 결정 + 환경 구축 — 2026-08-13. **FSGS 선택**(SparseGS 대비 별 2.4배(544 vs 229), 이슈 활동 더 많음(42 vs 9), 추가 무거운 의존성 없음(BoostingMonocularDepth 수동 다운로드가 필요한 SparseGS와 달리 MiDaS DPT-Hybrid를 torch.hub로 자동 다운로드), `--n_views` CLI로 연구 핵심 축(view 수)을 바로 통제 가능). 저장소는 `/data/Re-feem/code/fsgs`에 clone. **버전 어댑테이션**: 공식 environment.yml은 cudatoolkit 11.6/torch 1.12.1인데 우리 H200은 compute capability 9.0(Hopper) — MVSplat 때와 같은 이유로 torch 2.1.2+cu121로 교체. `setuptools`가 83.0.0으로 최신이라 `pkg_resources` 제거돼 있어 커스텀 CUDA 확장 빌드가 실패하는 것도 발견해 `setuptools<70`으로 다운그레이드해서 해결. `TORCH_CUDA_ARCH_LIST=9.0`으로 `simple-knn`/`diff-gaussian-rasterization-confidence` 둘 다 빌드 성공, import 검증 완료, `train.py --help`까지 완주(MiDaS DPT-Hybrid 470MB 체크포인트 자동 다운로드·캐시됨).
  - [x] **DTU 실제 학습 1회 완료** (2026-08-13). `analysis/prep_dtu_for_fsgs.py` 신규 — FSGS가 원래 요구하는 `colmap patch_match_stereo` dense point cloud를 우리 시스템엔 없는 COLMAP CLI 없이 우회: 우리 sparse triangulation 결과를 `{n_views}_views/dense/fused.ply` 자리에 대신 채움(Vanilla3DGS도 원래 sparse init이라 방법론적으로 무리 없음). 실행 중 발견한 이슈 2개: (1) FSGS가 LLFF 관례의 `poses_bounds.npy`(근/원 깊이 경계)도 요구 — MVSplat DTU config의 near=2.125/far=4.525 재사용해서 생성. (2) `--images` 기본값이 `images_8`(LLFF 8배 축소 관례)인데 우리 폴더명은 `images`라 빈 목록으로 실패 — `--images images` 명시로 해결. DTU scan1, seed0 12-view, 1000 iteration 학습 완료: test PSNR 12.81dB/SSIM 0.562/LPIPS 0.548(짧은 검증 실행이라 낮음, 정상 iteration 수는 아직 안 돌림)
  - [ ] **남은 일**: protocol_utils 스키마 준수하는 `fsgs_runner.py` 작성(지금은 1회성 검증 스크립트뿐), `model_registry.py` 등록, RE10K/DL3DV 데이터로도 같은 방식 확장
- [x] RE10K `.torch` chunk를 정식 runner 입력으로 연결 — 2026-08-12, `core/re10k_dataset.py::load_views()`가 vanilla_3dgs_runner.py/mvsplat_re10k_runner.py 양쪽에서 실사용 중
- [x] Vanilla3DGS/MVSplat을 RE10K 256x256에서 실제로 실행 — 2026-08-12 밤 완료. C1-b(warm-start) 20-scene × 4 view_count, C1-a 파일럿(일반 COLMAP init) 5-scene × 4 view_count, 둘 다 실측 로그 있음(아래 V3/C1-a 항목 참고)
- [x] densification on/off CLI 추가 — 2026-08-12, `vanilla_3dgs_runner.py`에 `--densification {on,off}` 추가(off는 `refine_stop_iter=0` 강제), `run_experiment_batch.py`에도 pass-through 배선. DTU scan1 4-view 90s 실측으로 검증: on은 gaussians 313→516→799→2120(30/60/90s), off는 끝까지 313 고정. 로그 파일명에 `_densoff` suffix를 붙여 on/off 결과가 서로 덮어쓰지 않게 함.
- [ ] co-visibility 기반 view selector 구현 — overlap 계산 도구(`generate_overlap.py`)는 있지만 "선택" 로직 자체가 없음(grep 0건)
- [x] **V3/C1-b 파이프라인(메커니즘) 완성 + DTU로 end-to-end 검증** (2026-08-12):
  - [x] FF(MVSplat/DepthSplat) Gaussian → gsplat 포맷 변환기(`core/ff_gaussian_convert.py`) — covariance 고유분해→scale/quat, opacity inverse-sigmoid, harmonics 재배열. 합성 데이터 round-trip으로 검증(covariance 재구성 오차 최대 2.6e-6)
  - [x] 렌더 등가성 gate 검사 함수(`analysis/check_renderer_equivalence.py`) — DTU scan1 2-view 실제 MVSplat 출력(13만 Gaussian)으로 held-out 7-view 재렌더링 비교, PSNR 35.6~42.0dB로 PASS. `overall.md` §5.8에 실측 근거로 tolerance 재정의(MSE 0.0001→PSNR≥33dB 제안, 파일럿 전 확정 필요)
  - [x] `vanilla_3dgs_runner.py`에 `--warm-start-checkpoint`/`--pose-scale-factor`/`--initial-sh-degree`/`--image-shape` 추가, `init_source="ff_warm_start"` 로깅. refinement=off(0s) baseline 평가 구현
  - [x] **해상도 불일치 블로커 해결**: `dtu_dataset.py`에 `resize_and_crop()`(MVSplat `crop_shim.py`와 동일 convention — Lanczos resize+center crop, cx/cy는 MVSplat과 동일하게 정중앙 강제) 추가, `load_scan(..., target_shape=)`로 연결. `--image-shape 256 256`으로 재실행하니 PSNR 9.20/SSIM 0.307/LPIPS 0.477 — MVSplat 자체 평가(9.25/0.308/0.480)와 노이즈 수준 오차로 일치
  - [x] refinement off(0s) vs on(5/10/20s) 실측(DTU scan1 2-view): off=9.20dB → on 5s=9.38 → 10s=9.39 → 20s=9.42dB, gaussian 수 131,072→164,131(densify 작동). **같은 초기값에서 refinement 효과가 실측으로 단조 증가함을 확인** — V3가 측정하려던 현상이 실제로 나옴
  - [x] **RE10K로 이식** (2026-08-12, 같은 세션) — `mvsplat_re10k_runner.py` 신규 작성(re10k_main_subset.json의 공식 context/target 재사용, DTU_SCALE_FACTOR 없이 raw RE10K pose 그대로), `re10k_dataset.py`에 `load_views()` 추가(dtu_dataset.load_scan과 동일 dict 형태 반환), `vanilla_3dgs_runner.py`에 `--dataset {dtu,re10k}` 분기 추가. RE10K main subset scene 1개(`0588138dfec165a1`, 2-view, official context=[70,160] — 앞서 overlap 분석에서 SfM 매칭 0건이었던 바로 그 wide-baseline 케이스)로 end-to-end 실행: refinement=off 기준 PSNR 17.253이 MVSplat 자체 평가(17.246)와 거의 일치(오차 0.007dB) → 변환·warm-start가 RE10K에서도 정확함을 확인
  - [x] **실측 결과 — 흥미로운 반전 신호**: 이 scene에서는 refinement가 품질을 **낮췄다**(off=17.25dB → on 5s=16.64 → 10s=16.62 → 20s=16.61dB). `oracle_checkpoint`도 정확히 iteration 0(=refinement 전)을 최고점으로 잡음. DTU 2-view(같은 세션 앞부분)에서는 반대로 refinement가 +0.22dB 개선시켰던 것과 대비됨 — overall.md의 사전 가설 **H3(초기 geometry 품질이 높으면 refinement 한계이득이 소멸/역전)**과 정확히 같은 방향의 첫 실측 신호. 다만 scene 1개·seed 1개 결과라 일반화 불가, main subset 20개 스케일로 반복해야 진짜 패턴인지 판단 가능
- [x] **V3/C1-b main subset 20 scene 전체 스케일업 완료** (2026-08-12, `batch/run_re10k_c1b_scaleup.py` 신규):
  - **버그 발견·수정**: 스케일업 중 일부 scene에서 refinement 도중 PSNR이 26→6.7dB로 폭락하는 걸 발견. 원인은 gsplat `DefaultStrategy`의 opacity reset(`reset_every=3000`, "처음부터 학습" 가정)이 짧은 warm-start 예산(60s, iter~3000 근처) 안에서 걸리면 좋은 FF 초기값을 파괴하고 복구를 못 하는 것 — `--reset-every 1000000`(사실상 비활성화)으로 C1-b warm-start 경로에 한해 고치고 재확인(재현/해소 둘 다 실측 확인)
  - **결과(off vs on 60s, view_count=2, 17/20 scene 유효)**: 6개 개선(+0.11~+1.60dB), 11개 소폭 하락(-0.06~-1.84dB), 평균 delta **-0.14dB**(대체로 무승부에 가까움, scene마다 방향이 갈림). 3개 scene은 렌더 등가성 gate가 근소하게 미달(28.98~33.14dB, PSNR≥33dB 기준 대비)해서 refinement 자체를 스킵함 — DTU 1개 scene으로 잡았던 tolerance가 RE10K 20개 스케일에서는 경계선 케이스가 나온다는 뜻, 최종 tolerance 재검토 필요
  - 원본 데이터: `experiments/outputs/re10k_c1b_scaleup/c1b_scaleup_summary_full20.json`
  - [x] **4/8/12-view까지 전부 완료** (2026-08-12 저녁). view_count별 결과(off vs on 60s, 20 scene):

    | view_count | gate 통과 | 개선 | 하락 | 평균 delta |
    |---:|---:|---:|---:|---:|
    | 2 | 17/20 | 6 | 11 | -0.14dB |
    | 4 | 20/20 | 18 | 2 | +3.67dB |
    | 8 | 20/20 | 20 | 0 | +7.71dB |
    | 12 | 20/20 | 20 | 0 | +10.47dB |

    view 수가 늘수록 refinement 효과가 커지는 뚜렷한 단조 증가 패턴. **다만 중요한 교란요인이 있다**: §5.2에서 이미 확인했듯 MVSplat은 2-view 전용 학습이라 4/8/12-view는 분포 밖 사용이다. 실제로 off(=MVSplat 단독) 평균 PSNR이 2-view 25.4→4-view 20.2→8-view 19.0으로 **view가 늘수록 오히려 나빠진다**(분포 밖이라 혼란스러워하는 것으로 해석). 즉 "12-view에서 refinement가 +10.5dB나 개선"은 순수하게 "view가 많을수록 refinement가 좋다"가 아니라 상당 부분 **"MVSplat이 분포 밖에서 만든 나쁜 초기값을 gradient 기반 refinement가 복구·역전시킨 효과"**로 봐야 한다. 그래도 흥미로운 점: on_60s 절대값 자체도 view가 늘수록 계속 좋아진다(25.2→23.9→26.7→29.9dB) — refinement가 "나쁜 초기값 복구"만 하는 게 아니라 view 수가 늘면서 생기는 추가 정보(더 많은 photometric constraint)까지 실제로 활용한다는 뜻. 원본: `experiments/outputs/re10k_c1b_all_viewcounts_summary.json`
  - [x] **DepthSplat을 C1-b에 연결** (2026-08-12) — MVSplat OOD 교란요인을 분리하기 위한 작업. 로컬에 있는 DepthSplat 체크포인트가 DL3DV 전용(`depthsplat-gs-base-dl3dv-256x448-randview2-6`, RE10K+DL3DV 혼합 체크포인트는 아직 미다운로드)이라 RE10K 대신 DL3DV로 진행(오늘 이미 만든 DL3DV overlap 인프라 재사용). 신규: `runners/depthsplat_dl3dv_runner.py`(MVSplat용과 같은 gaussians.pt/render_reference.pt 포맷 저장), `core/dl3dv_dataset.py::load_views()`에 `target_shape` 리사이즈 지원 추가, `vanilla_3dgs_runner.py`에 `--dataset dl3dv` 분기 추가. DL3DV pilot 1 scene(2-view)으로 end-to-end 검증: 렌더 등가성 gate 43~54dB로 PASS(MVSplat의 35~42dB보다도 정밀), warm-start baseline PSNR 9.49dB가 DepthSplat 자체 평가(9.51dB)와 거의 일치. refinement 20s 후 8.94dB로 소폭 하락 — 이 1개 scene만으로는 아직 결론 못 냄
  - [x] **DepthSplat C1-b 25-scene 스케일업 완료, 교란요인 가설 확인됨** (2026-08-12, `run_dl3dv_c1b_scaleup.py`). 2/4-view(DepthSplat 분포 안) 결과:

    | 모델/데이터셋 | 2-view delta | 4-view delta | off 평균(2→4-view) |
    |---|---:|---:|---|
    | MVSplat/RE10K(4-view는 분포 밖) | -0.14dB | **+3.67dB** | 25.4→**20.2dB**(나빠짐) |
    | DepthSplat/DL3DV(둘 다 분포 안) | +0.10dB | +0.15dB | 10.9→**17.0dB**(좋아짐) |

    MVSplat은 2→4-view에서 refinement 효과가 폭발적으로 커지는데(-0.14→+3.67) DepthSplat(분포 안)은 거의 그대로다(+0.10→+0.15). 결정적으로 off(FF 단독) 점수 자체가 MVSplat은 view가 늘수록 나빠지고(분포 밖이라 혼란) DepthSplat은 좋아진다(정상). **결론: 아침에 나온 "view 수가 늘수록 refinement 효과가 커진다"는 현상은 view-count 자체의 효과가 아니라 MVSplat이 분포 밖에서 무너지는 걸 refinement가 복구하는 효과였다는 가설이 확인됨.** 원본: `experiments/outputs/dl3dv_c1b_scaleup_{2,4}view/c1b_scaleup_summary.json`
    - **한계(정직하게 기록)**: 모델도 다르고(MVSplat vs DepthSplat) 데이터셋도 다름(RE10K vs DL3DV), 게다가 이 스케일업은 DL3DV view 선택 v1(방금 v2로 고친 것 이전 버전)로 돌아간 것 — 완벽한 단일 변수 통제 비교는 아님. v2로 재실행하면 더 깨끗한 비교가 될 것
  - [x] **renderer_equivalence_tolerance 최종 동결** (2026-08-13) — 누적 gate 로그 130건(view-PSNR 샘플 390개, MVSplat/RE10K 240 + DepthSplat/DL3DV 150)을 모아 재검토, **PSNR≥33dB 유지 확정**(미달 2.3%뿐, 재구성 품질 범위 8~25dB대와 충분히 구분됨). `overall.md` §5.8, `experiment_config.yaml`(`renderer_equivalence_tolerance_psnr_db: 33.0`) 반영
  - [ ] **남은 일**: DL3DV v2(고친 view 선택)로 DepthSplat C1-b 재실행해서 더 깨끗한 비교 확보. RE10K 일반 COLMAP/random-init 경로는 연결됐지만(§C1-a) 아직 main subset 전체 스케일은 아님

## 검증 결과

- [x] unit test 9개 통과 (`python3 -m unittest discover -s tests`)
- [x] manifest 12,480 row 생성 (main 7,680 + c1b 3,840 + c2 960)
- [x] DTU dense-view sanity (scan1, 42-view, 30k iter, PSNR 24.0~24.1dB)
- [x] MVSplat RE10K in-domain (25.6dB / mirror chunk 22.4dB)
- [x] MVSplat DTU zero-shot (4.8~11.8dB, OOD로 해석)
- [x] DepthSplat DL3DV in-domain (2-view, 20.0dB)
- [x] DTU scan1 2/4/8/12-view 통제 스모크 (2026-08-11, 위 항목과 동일 근거)
- [x] RE10K에서의 동일한 2/4/8/12-view 통제 스모크 — 2026-08-12, C1-b warm-start 경로로 2-view는 20 scene 전부, 4-view도 진행 중(8/12-view 대기). MVSplat "일반"(non-warm-start) 경로는 아직 없음(별도 항목)
- [ ] DL3DV에서의 동일한 통제 스모크 — overlap 계산까지만 완료(아래), Vanilla3DGS/MVSplat 실행은 아직 없음
- [x] RE10K/DL3DV용 overlap bucket (low/high threshold) 산출 — 위 "모델/러너" 섹션 참고(중복 추적)

## 분석 / 이론

- [x] Gauss-Newton / J^T J 수식 정리 문서 (`experiments/docs/paper/paper_gauss_newton_notation.md`)
- [x] geometry uncertainty figure 1차 생성 + 실측 (DTU scan1, 861 pair, `experiments/outputs/geometry_figures/pairwise_geometry.csv`)
- [x] overlap-uncertainty 부호 이상 현상 발견 및 원인 분석(baseline confound)
- [x] 위 부호 이상 현상 재해석 완료 (2026-08-12, A-1 완료 후 사용자와 직접 분석). raw corr +0.952 → baseline 선형 통제해도 +0.801(불충분) → shared_points 가설 세워서 검증했으나 기각(+0.071, 무관) → baseline이 log-log 관계(power-law)임을 확인하고 log(baseline)로 통제하니 +0.301로 크게 감소. 결론: 원래 가설(baseline 교란)이 맞았고, 처음 선형 통제가 함수형을 잘못 잡아서 가짜 잔차가 남았던 것. 전체 유도·해석·논문 초안 문장: `paper/paper_geometry_confound_analysis_2026-08-12.md`, 재현 코드는 `geometry_uncertainty_figure.py::print_confound_analysis()`에 통합
- [ ] RE10K/DL3DV(경로형 카메라)에서도 같은 baseline-overlap 관계가 성립하는지 확인 — DTU(orbital rig)와 다를 수 있음
- [ ] co-visibility 기반 view selector 구현 (모델/러너 섹션과 중복 추적)
- [ ] overlap bucket threshold를 본 실험용으로 동결

---

## 진행률 감각 (검증 근거 기준, 감이 아니라 위 체크박스 비율)

| 구간 | 대략 진행률 |
|---|---:|
| 연구 설계 / 프로토콜 | ~85% (핵심 12항목 중 미결 3개: C2 budget, 조기종료 규칙, 통계계획 동결) |
| 데이터 확보 | ~80% |
| 모델 / 러너 | ~60% (V3/C1-b가 DTU+RE10K 20-scene까지 실측 완료됐지만 4/8/12-view·DepthSplat·RE10K 일반 경로는 남음) |
| 검증 결과 | ~60% (DTU는 두텁게 검증, RE10K는 C1-b 20-scene까지 붙음, DL3DV는 아직 얇음) |
| 분석 / 이론 | ~60% |

논문에 쓸 결과 생산 이전, 파일럿 직전 단계라는 평가는 유효함.

## 다음 우선순위 (2026-08-12 저녁 기준, 1~4 완료로 갱신)

1. ~~densification on/off CLI~~ ✅
2. ~~RE10K main subset index + overlap bucket~~ ✅
3. ~~V3(C1-b) 구현 + DTU/RE10K 20-scene 스케일업 (2/4/8/12-view 전부)~~ ✅
4. ~~C2 budget 결정~~ ✅
5. ~~DL3DV overlap 이식~~ ✅ (view 선택 방식 한계 있음, 아래 6번)
6. ~~DepthSplat을 C1-b 파이프라인에 연결~~ ✅ (2026-08-12, 1개 scene으로 end-to-end 검증. 아래 새 항목으로 이어짐)
7. ~~DL3DV view 선택을 DepthSplat 방식으로 재실행~~ ✅ (2026-08-12) — 가설 확인됨. 4-view zero-overlap 56%(14/25)→4%(1/25), median 0.000→0.615. 8/12-view도 개선. 2-view는 여전히 25개 전부 0(다른 데이터셋과 동일, 진짜 현상). 결과: `experiments/outputs/dl3dv_overlap_v2/`(기존 `dl3dv_overlap/`은 다른 작업이 읽고 있어 보존) — 앞으로 DL3DV는 v2 사용
8. ~~Vanilla3DGS "일반"(non-warm-start) 경로를 RE10K/DL3DV에 연결~~ ✅ (2026-08-12 밤) — C1-a(진짜 Regime Map)로 가는 마지막 관문. `vanilla_3dgs_runner.py`에 `_colmap_init_from_loaded_views()` 추가(이미 메모리에 로드된 RE10K/DL3DV view를 임시 디렉토리에 써서 known-pose COLMAP triangulation — `generate_re10k_view_overlap.py`류와 같은 공용 코어 재사용), `--dataset re10k/dl3dv`의 "warm-start 필수" 제약 제거. 실측 검증: RE10K 12-view에서 `init_source=colmap_sfm`(593 point)으로 15초 만에 20.9→28.4dB로 정상 학습(4-view는 SfM point 부족으로 random_sphere_fallback, 이것도 의도된 정상 동작). DL3DV 8-view도 동작 확인(fallback 경로). MVSplat 쪽은 `mvsplat_re10k_runner.py`가 이미 이 역할(추론 전용, C1-b 아닌 C1-a용)을 하고 있어 추가 작업 없음 — **이제 RE10K에서 두 패러다임(Vanilla3DGS optimization / MVSplat feed-forward)을 같은 scene·view에서 다 돌릴 수 있어 C1-a 착수 가능**
9. ~~C1-a 첫 파일럿 실행~~ ✅ (2026-08-12 밤, `run_re10k_c1a_pilot.py`) — RE10K 5 scene × 2/4/8/12-view, budget[1,10,60]s, Vanilla3DGS(일반 COLMAP init) vs MVSplat 정면 비교. **역전 패턴 확인됨**: 2/4-view는 예산과 무관하게 MVSplat 압승(21.7/16.3dB vs 7~11dB). 8-view부터 10초만 줘도 Vanilla3DGS가 역전(16.8 vs 16.5dB). 12-view/60s에서는 Vanilla3DGS가 크게 앞섬(20.6 vs 17.1dB). §2 연구 질문("역전 경계가 무엇으로 결정되는가")에 대한 첫 실측 증거 — view 수와 시간 예산이 함께 늘 때 optimization이 이긴다. **주의**: seed 1회·scene 5개뿐이라 파일럿 단계, 통계적 결론 아님(4-view Vanilla3DGS가 2-view보다 낮게 나온 것도 표본이 적어서일 가능성). 원본: `experiments/outputs/re10k_c1a_pilot/c1a_pilot_summary.json`
10. ~~renderer_equivalence_tolerance 최종 동결~~ ✅ (2026-08-13, PSNR≥33dB 유지 확정)
11. co-visibility selector 연결
12. C1-a를 main subset 20 scene × seed 3회로 스케일업(진짜 본 실험)
