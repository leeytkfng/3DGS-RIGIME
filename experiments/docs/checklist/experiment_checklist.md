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
- [x] §5.2 모델별 지원 view 수 표 완성 — 2026-08-12, MVSplat/DepthSplat 공식 config·README 기준으로 채움, `model_registry.py`의 `supports_views`도 동기화(MVSplat→`[2]`, DepthSplat→`[2,4]`). **아직 미커밋** (`git status`: overall.md, model_registry.py modified)
- [x] §5.4 GPU-hour 예산 재계산 — 2026-08-12, manifest trajectory 수 × 실측 wall-clock 기반으로 재계산(약 186~250 GPU-hour). **아직 미커밋**
- [ ] confidence 출력 열 — MVSplat/DepthSplat 둘 다 confidence/uncertainty 출력이 소스코드에 없음을 확인(grep 결과 0건). 표에는 "없음"으로 채워졌으나, 논문에서 이걸 어떻게 다룰지(대체 지표 필요한지) 결정 안 됨
- [ ] C2 depth noise/scale bias의 **budget_seconds가 manifest에 아예 없음** — 2026-08-12 발견. 60s 가정 시 GPU-hour 18.1, 300s 가정 시 82.1로 4배 차이 남. 파일럿 전 필수 동결 항목
- [ ] §5.11 조기 종료 규칙 동결 (미결 항목으로 문서에 이미 표시돼 있음)
- [ ] §5.12 통계 분석 계획 최종 동결

## 데이터

- [x] `/data/Re-feem` 디렉토리 구성
- [x] DTU 공식 split 16 scan + extra 13 scan = 29 scan
- [x] RE10K test 114 scene (probe 41 + mirror 73)
- [x] DL3DV pilot 25 scene, 공식 benchmark split과 중복 0 확인
- [x] RE10K/DL3DV `SOURCE.md` 작성
- [x] RE10K main subset(20~30 scene) index 생성 — 2026-08-12, `generate_re10k_main_subset.py` 작성·실행. 로컬 114 scene ∩ MVSplat 공식 evaluation index(`assets/evaluation_index_re10k.json`, non-null 6,474건) ∩ frame수≥50 ∩ context/target 안 겹침 조건으로 96개 후보 중 seed=0으로 20개 결정론적 선정. 2-view는 공식 context/target 그대로, 4/8/12-view는 target(3-view, view_count 불문 고정)을 제외한 pool에서 seeded 생성. 출력: `experiments/outputs/re10k_main_subset/re10k_main_subset.json`. 검증: 20 scene × 4 view_count 전부 context/target 겹침 없음, index 범위 안(스크립트로 재확인). **부수 발견**: 공식 index 자체에 context/target이 겹치는 scene이 2개 있었음(`aadc1e2dc74fd644`, `cdf439b17a6a98d4`) — leakage 방지를 위해 main subset에서 제외
- [x] RE10K 2/4/8/12-view candidate에 대한 overlap 계산 — 2026-08-12. `colmap_init.py`를 DTU 전용에서 데이터셋 무관 공용 코어(`triangulate_sfm_points_from_cameras`)로 리팩터(DTU 경로는 얇은 wrapper로 남겨 기존 동작 그대로 유지 — scan1 4-view 313 SfM point로 재검증). RE10K 전용 로더 `core/re10k_dataset.py`(`.torch` chunk에서 필요한 frame만 디스크로 풀고 정규화된 카메라를 픽셀 K로 변환) + `analysis/generate_re10k_view_overlap.py` 신규 작성. main subset 20 scene × 4 view_count = 80 combo 전부 COLMAP 실행 완료(`experiments/outputs/re10k_main_subset/overlap/all_scenes_summary.json`). **핵심 발견**: 2-view는 20 scene 전부 mean_overlap=0.000(SfM 매칭 0건) — DTU에서 봤던 "2-view SfM 붕괴"가 RE10K에서도 100% 재현됨(MVSplat 공식 2-view context가 SfM 매칭 목적이 아니라 wide-baseline NVS 목적으로 뽑혀서 그런 것으로 추정). 4/8/12-view는 정상 범위(median 0.80/0.55/0.52)
- [x] RE10K/DL3DV용 overlap bucket (low/high threshold) 산출 — RE10K는 위 실측으로 완료(§5.3 stratify_thresholds_within_view_count 원칙대로 view_count별 median split): 4-view 0.804, 8-view 0.552, 12-view 0.524. 2-view는 분리 불가(전부 0). 결과: `experiments/outputs/re10k_main_subset/overlap/bucket_thresholds.json`. **DL3DV는 아직 미착수**
- [ ] RE10K citation/license 문구 논문용 정리

## 모델 / 러너

- [x] Vanilla3DGS runner (COLMAP init, LPIPS, densification 궤적 로깅) — `experiments/scripts/runners/vanilla_3dgs_runner.py`
- [x] MVSplat 정식 runner (protocol_utils 스키마) — `experiments/scripts/runners/mvsplat_runner.py`
- [x] DepthSplat probe (정식 runner 아님) — `experiments/scripts/probes/depthsplat_dl3dv_probe.py`
- [x] model_registry 구조 + batch driver 일반화
- [x] DTU 공식 split 16 scan x 2-view x seed0 batch
- [x] DTU scan1 2/4/8/12-view 통제 스모크 — Vanilla3DGS/MVSplat 둘 다 8개 run 전부 `status: ok` (`experiments/outputs_smoke_20260811/logs/batch_summary.json` 직접 확인)
- [ ] DepthSplat probe -> 정식 runner 승격 (elapsed/wall_clock 로깅도 없어서 §5.4 GPU-hour 추정에 DepthSplat 실측치를 못 넣었음 — 승격 시 같이 고칠 것)
- [ ] SparseGS/FSGS 통합 — 코드 자체가 없음(`/data/Re-feem/code`에 sparse/fsgs 디렉토리 없음, 확인함)
- [ ] RE10K `.torch` chunk를 정식 runner 입력으로 연결
- [ ] Vanilla3DGS/MVSplat을 RE10K 256x256에서 실제로 실행 — outputs 디렉토리에 RE10K 관련 로그 0건(확인함)
- [x] densification on/off CLI 추가 — 2026-08-12, `vanilla_3dgs_runner.py`에 `--densification {on,off}` 추가(off는 `refine_stop_iter=0` 강제), `run_experiment_batch.py`에도 pass-through 배선. DTU scan1 4-view 90s 실측으로 검증: on은 gaussians 313→516→799→2120(30/60/90s), off는 끝까지 313 고정. 로그 파일명에 `_densoff` suffix를 붙여 on/off 결과가 서로 덮어쓰지 않게 함. **아직 미커밋**
- [ ] co-visibility 기반 view selector 구현 — overlap 계산 도구(`generate_overlap.py`)는 있지만 "선택" 로직 자체가 없음(grep 0건)
- [ ] **V3/C1-b 실제 구현** (2026-08-12 새로 식별된 항목, 이전 체크리스트에 없었음):
  - [ ] FF(MVSplat/DepthSplat) Gaussian 출력 -> standard 3DGS(gsplat) 포맷 변환기
  - [ ] 렌더 등가성 gate 검사 함수 (지금은 config에 tolerance 숫자만 있고 이걸 읽어 비교하는 코드가 없음)
  - [ ] `vanilla_3dgs_runner.py`의 `init_gaussians()`에 FF warm-start 경로 추가 (현재는 `colmap_sfm`/`random_sphere_fallback` 둘뿐)
  - [ ] refinement off(0s) vs on(10/60/300s) 비교 루프

## 검증 결과

- [x] unit test 9개 통과 (`python3 -m unittest discover -s tests`)
- [x] manifest 12,480 row 생성 (main 7,680 + c1b 3,840 + c2 960)
- [x] DTU dense-view sanity (scan1, 42-view, 30k iter, PSNR 24.0~24.1dB)
- [x] MVSplat RE10K in-domain (25.6dB / mirror chunk 22.4dB)
- [x] MVSplat DTU zero-shot (4.8~11.8dB, OOD로 해석)
- [x] DepthSplat DL3DV in-domain (2-view, 20.0dB)
- [x] DTU scan1 2/4/8/12-view 통제 스모크 (2026-08-11, 위 항목과 동일 근거)
- [ ] RE10K에서의 동일한 2/4/8/12-view 통제 스모크 — 로그 없음(확인함)
- [ ] DL3DV에서의 동일한 통제 스모크 — 로그 없음(확인함)
- [ ] RE10K/DL3DV용 overlap bucket (low/high threshold) 산출

## 분석 / 이론

- [x] Gauss-Newton / J^T J 수식 정리 문서 (`experiments/docs/paper/paper_gauss_newton_notation.md`)
- [x] geometry uncertainty figure 1차 생성 + 실측 (DTU scan1, 861 pair, `experiments/outputs/geometry_figures/pairwise_geometry.csv`)
- [x] overlap-uncertainty 부호 이상 현상 발견 및 원인 분석(baseline confound)
- [ ] 위 부호 이상 현상 재해석 — A-1(Gauss-Newton) 완료 후로 보류 중
- [ ] co-visibility 기반 view selector 구현 (모델/러너 섹션과 중복 추적)
- [ ] overlap bucket threshold를 본 실험용으로 동결

---

## 진행률 감각 (검증 근거 기준, 감이 아니라 위 체크박스 비율)

| 구간 | 대략 진행률 |
|---|---:|
| 연구 설계 / 프로토콜 | ~85% (핵심 12항목 중 미결 3개: C2 budget, 조기종료 규칙, 통계계획 동결) |
| 데이터 확보 | ~80% |
| 모델 / 러너 | ~50% (V3/C1-b 구현이 새 항목으로 추가되며 체감 진행률 하락) |
| 검증 결과 | ~55% (DTU는 두텁게 검증, RE10K/DL3DV는 아직 얇음) |
| 분석 / 이론 | ~60% |

논문에 쓸 결과 생산 이전, 파일럿 직전 단계라는 평가는 유효함.

## 다음 우선순위 (2026-08-12 기준 합의된 순서)

1. densification on/off CLI 추가 — 제일 싸게 끝남
2. RE10K main subset index + 2/4/8/12-view candidate + overlap bucket — DTU 패턴 이식, V1/V3 둘 다의 선행조건
3. V3(C1-b) 구현 — FF→3DGS 변환기, 렌더 등가성 gate, warm-start, refinement loop
4. C2 budget 결정 (§5.4 GPU-hour 확정의 유일한 미결 변수)
5. DepthSplat 정식 승격, co-visibility selector 연결, SOURCE.md/표 정리
