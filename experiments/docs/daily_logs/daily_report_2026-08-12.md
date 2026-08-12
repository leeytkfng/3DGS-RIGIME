# 일일 보고서 — 2026-08-12

## 오늘 목표

체크리스트(`experiments/docs/checklist/experiment_checklist.md`)에서 합의한 우선순위대로: (1) §5.2/§5.4 미결 항목 채우기, (2) densification on/off CLI 추가, (3) RE10K main subset 20-scene index 생성, (4) 그 subset의 2/4/8/12-view candidate에 DTU와 같은 방식으로 overlap 계산. 그리고 어제부터 막혀있던 git push 재확인.

---

## 1. git push 재확인

원인은 용량이 아니라 **인증 정보가 아예 없는 것**이었다(`credential.helper` 미설정, `gh` 미설치, 토큰 없음). 사용자가 PAT를 발급해줘서 remote URL에 일회성으로 넣어 push하고 즉시 토큰 없는 URL로 되돌렸다. push 성공(`a848ec5..14f117a`). 채팅에 토큰이 노출됐으므로 revoke 권고함.

## 2. §5.2 모델별 지원 view 수 표 완성

`overall.md`의 "확인 필요" placeholder를 실제 공식 config/README로 채웠다.

- **MVSplat**: RE10K 학습이 고정 2-view(`config/dataset/view_sampler/bounded.yaml`), DTU 공식 eval index도 N=2,3만 제공, README가 직접 "12-view는 DepthSplat 쓰라"고 안내 — 저자 스스로도 4-view 이상은 범위 밖으로 취급.
- **DepthSplat**: 우리가 쓰는 체크포인트(`randview2-6`)가 실제로 2~6-view 랜덤 샘플링으로 학습됨을 config(`view_sampler/boundedv2_360.yaml`)로 재확인. 8/12-view는 별도 체크포인트(`randview4-10`, 448×768)가 필요한데 아직 미보유.
- `model_registry.py`의 `supports_views`(지금까지 네 모델 전부 `[2,4,8,12]` 동일 placeholder)를 MVSplat→`[2]`, DepthSplat→`[2,4]`로 갱신. manifest 재생성 시 `validate_config()`가 의도대로 경고를 냈다: `MVSplat does not list support for views: [4, 8, 12]`.

## 3. §5.4 GPU-hour 재계산

manifest 12,480 row를 budget-checkpoint(=한 trajectory 안의 스냅샷) 기준으로 접어서 실행 단위(2,880개)를 다시 세고, 어제 DTU smoke 실측 wall-clock을 대입했다.

| 구간 | trajectory 수 | per-trajectory | 소계 |
|---|---:|---:|---:|
| main optimization | 960 | ~308s(COLMAP 오버헤드 8.15s + 300s) | 82.1h |
| main feed-forward | 960 | ~7s(MVSplat 실측) | 1.9h |
| C1-b refinement-on | 960 | ~308s | 82.1h |
| C1-b refinement-off | 960 | ~7.5s(평가만) | 2.0h |
| C2 | 960 | **budget_seconds가 manifest에 아예 없음** | 18.1~82.1h |

**합계 약 186~250 GPU-hour** — 8/9 audit이 걱정했던 "2.4배 과소추정"은 해소됐다(budget을 trajectory로 접어 세면 원래 추정이 맞았음). 대신 **C2의 budget이 protocol에 정의돼 있지 않다는 것**이 새로 드러난 진짜 병목(4배 차이의 유일한 변수) — 파일럿 전 동결 항목으로 추가.

## 4. densification on/off CLI

`vanilla_3dgs_runner.py`에 `--densification {on,off}` 추가(off는 gsplat `DefaultStrategy`의 `refine_stop_iter=0` 강제). `run_experiment_batch.py`에도 pass-through 배선, 로그/체크포인트 경로에 `_densoff` suffix를 붙여 on/off 결과가 같은 파일을 안 덮어쓰게 했다.

**실측 검증**(DTU scan1, 4-view, `--refine-start-iter 10` 임시 오버라이드로 densify가 iteration 안에 들어오게 함):

| 조건 | 30s | 60s | 90s |
|---|---:|---:|---:|
| on | gaussians=516 | 799 | 2120 |
| off | gaussians=313(고정) | 313 | 313 |

의도대로 정확히 동작.

## 5. RE10K main subset 20-scene index 생성

`generate_re10k_main_subset.py` 신규 작성. 자체 선택이 아니라 MVSplat/DepthSplat이 실제로 쓰는 **공식 RE10K evaluation index**(`assets/evaluation_index_re10k.json`, 7,194 scene 중 non-null 6,474개)와 우리 로컬 114 scene의 교집합에서 뽑았다. 2-view는 공식 context/target을 그대로 쓰고, 4/8/12-view는 공식 index가 제공하지 않아 DTU smoke와 같은 seeded 규칙으로 생성했으며, target(3-view held-out)은 view 수와 무관하게 고정했다.

**부수 발견**: 공식 index 자체에 context/target 프레임이 겹치는 scene이 2개 있었다(`aadc1e2dc74fd644`, `cdf439b17a6a98d4`). 우리 §5.7 leakage 방지 원칙과 충돌해 main subset에서 제외(96개 후보 중 20개 확정). 출력: `experiments/outputs/re10k_main_subset/re10k_main_subset.json`.

## 6. RE10K 2/4/8/12-view overlap 계산 — DTU 방식 이식

`colmap_init.py`를 DTU 전용 함수에서 **데이터셋 무관 공용 코어**(`triangulate_sfm_points_from_cameras`)로 리팩터했다. DTU 쪽은 얇은 wrapper로 남겨 기존 signature/동작을 그대로 유지했고, 리팩터 직후 DTU scan1 4-view를 재실행해 SfM point 수(313)가 리팩터 전과 정확히 같음을 확인했다.

RE10K 전용 로더 `core/re10k_dataset.py`를 새로 작성했다 — `.torch` chunk에서 필요한 frame만 디스크에 풀고, MVSplat 코드(`convert_poses`)로 재확인한 카메라 규약(정규화된 fx/fy/cx/cy를 픽셀 단위로 스케일, w2c 3x4를 별도 inverse 없이 그대로 R/t로 사용)에 맞춰 COLMAP 입력을 만든다. `generate_re10k_view_overlap.py`로 main subset 20 scene × 4 view_count = 80 combo를 전부 실행했다.

**핵심 발견**: 20 scene 전부 2-view에서 mean_overlap=0.000(SfM 매칭 0건). DTU에서 봤던 "2-view SfM 붕괴"가 RE10K에서도 예외 없이 재현됐다 — MVSplat 공식 2-view context가 SfM 재구성이 아니라 wide-baseline novel-view-synthesis 목적으로 뽑히기 때문으로 보인다. 4/8/12-view는 정상 범위(view_count 내부 median: 0.804 / 0.552 / 0.524)였고, 이 median으로 §5.3 stratify 원칙에 따라 low/high overlap bucket 경계를 잡았다(`bucket_thresholds.json`).

## 7. V3(C1-b) 파이프라인 구축 + DTU→RE10K 이식

당초 §5.2/§5.4/데이터 작업 이후로 미뤄뒀던 V3(동일 FF 초기값에서 refinement on/off 순효과 측정)를 오늘 안에 끝까지 만들었다.

**부품 3개, 각각 검증:**
- `core/ff_gaussian_convert.py` — MVSplat `Gaussians`(means/covariances/harmonics/opacities, 월드좌표)를 gsplat 파라미터화로 변환. covariance 고유분해→scale/quaternion 변환은 합성 데이터 round-trip으로 먼저 검증(재구성 오차 최대 2.6e-6).
- `analysis/check_renderer_equivalence.py` — §5.8 렌더 등가성 gate. 두 conda env(mvsplat/ps3)가 호환 안 돼서 `mvsplat_runner.py`가 저장한 `gaussians.pt`/`render_reference.pt`로 cross-env hand-off. DTU 실측 PSNR 35.6~42.0dB로 PASS. **overall.md §5.8을 실측 근거로 갱신**: config의 `renderer_equivalence_tolerance: 0.0001`은 서로 다른 두 정상 CUDA rasterizer 간 흔한 수치오차보다도 타이트한 placeholder였음을 확인, PSNR≥33dB 기준을 제안.
- `vanilla_3dgs_runner.py --warm-start-checkpoint` — FF Gaussian을 그대로 최적화 시작점으로 로드. 처음엔 PSNR이 MVSplat 원본과 안 맞았는데(7.40 vs 9.25dB), 원인이 **해상도 불일치**(러너가 DTU 네이티브 1600×1200로 렌더링, FF Gaussian은 MVSplat 256×256 기준)임을 찾아 `dtu_dataset.py`에 `resize_and_crop()`(MVSplat crop_shim과 동일 convention) 추가, `--image-shape 256 256`으로 재실행하니 9.20dB로 일치(오차 0.05dB, gate가 예측한 노이즈 수준 그대로).

**DTU 2-view 실측**: off=9.20dB → on 5s=9.38 → 10s=9.39 → 20s=9.42dB(+0.22dB, gaussian 131,072→164,131). 같은 초기값에서 refinement가 실제로 개선시켰다.

**RE10K로 이식**: `mvsplat_re10k_runner.py` 신규 작성(re10k_main_subset.json의 공식 context/target 재사용, 좌표계 스케일 보정 불필요), `re10k_dataset.py`에 `load_views()` 추가, `vanilla_3dgs_runner.py`에 `--dataset {dtu,re10k}` 분기. main subset scene `0588138dfec165a1`(2-view, official context=[70,160] — 어제 overlap 분석에서 SfM 매칭 0건이었던 바로 그 wide-baseline scene)로 실행:

- refinement=off 기준 PSNR 17.253 — MVSplat 자체 평가(17.246)와 거의 일치. 변환·warm-start가 RE10K에서도 정확함을 재확인.
- **반전 신호**: off=17.25dB → on 5s=16.64 → 10s=16.62 → 20s=16.61dB — **refinement가 품질을 낮췄다.** `oracle_checkpoint`도 iteration 0을 최고점으로 잡음. DTU에서는 +0.22dB 개선, RE10K 이 scene에서는 반대로 악화 — overall.md 사전가설 **H3**(초기 geometry 품질이 높으면 refinement 한계이득이 소멸/역전)과 같은 방향의 첫 실측 신호다. scene 1개·seed 1개라 아직 일반화는 안 되고, main subset 20개로 스케일업해야 패턴인지 우연인지 판단 가능.

## 8. V3(C1-b) main subset 20 scene 전체 스케일업

RE10K scene 1개짜리 증명(§7)을 `batch/run_re10k_c1b_scaleup.py`로 20개 전체로 확장했다(2-view, off vs on 10s/60s).

**스케일업 도중 버그 발견·수정**: 몇몇 scene에서 refinement 도중 PSNR이 26dB대에서 6.7dB로 순간 폭락하는 걸 발견. train_loss가 같은 지점에서 0.006→0.28로 튀는 것까지 확인해 원인을 좁혔다 — gsplat `DefaultStrategy`의 opacity reset(기본 `reset_every=3000`)이 "처음부터 학습"을 가정한 안전장치인데, 짧은 warm-start 예산(60s ≈ 5000 iter) 안에서 iter~3000 근처에 걸리면 이미 좋은 FF 초기값의 opacity를 전부 날려버리고 남은 iteration으로 복구를 못 한다. `--reset-every 1000000`(사실상 비활성화)으로 재현/해소를 각각 실측 확인하고 C1-b warm-start 경로에 고정 적용했다.

**결과 (17/20 scene 유효, 3개는 렌더 등가성 gate 근소 미달로 스킵):**

| 방향 | scene 수 | delta 범위 |
|---|---:|---|
| 개선 | 6 | +0.11 ~ +1.60dB |
| 하락 | 11 | -0.06 ~ -1.84dB |

평균 delta **-0.14dB** — 대체로 무승부에 가깝고, scene마다 방향이 갈린다. 3개 gate 실패 scene의 PSNR은 28.98~33.14dB로, DTU 1개 scene으로 잡았던 tolerance(PSNR≥33dB)가 20-scene 스케일에서는 경계선에 걸리는 경우가 있다는 뜻 — 최종 tolerance는 이 20-scene 데이터까지 반영해서 재검토해야 한다. 원본: `experiments/outputs/re10k_c1b_scaleup/c1b_scaleup_summary_full20.json`.

## 9. C2 budget 결정 + V3 4/8/12-view 완주 + DL3DV overlap 이식 (저녁 세션)

**C2 budget**: manifest에 아예 없던 C2 예산을 main phase/C1-b와 동일한 300s trajectory + `[1,10,60,300]` 체크포인트로 확정(`overall.md` §5.9, `experiment_config.yaml`, `run_experiment.py` 반영). §5.4 GPU-hour 추정을 186~250h 범위에서 **250h로 확정**.

**V3(C1-b) 4/8/12-view 완주**: `off vs on(60s)` 결과, 20 scene 기준.

| view_count | gate 통과 | 개선 | 하락 | 평균 delta |
|---:|---:|---:|---:|---:|
| 2 | 17/20 | 6 | 11 | -0.14dB |
| 4 | 20/20 | 18 | 2 | +3.67dB |
| 8 | 20/20 | 20 | 0 | +7.71dB |
| 12 | 20/20 | 20 | 0 | +10.47dB |

view 수가 늘수록 refinement 효과가 커지는 뚜렷한 패턴이 나왔지만, **바로 결론 내리면 안 되는 이유가 있다**: MVSplat은 2-view 전용 학습이라(§5.2) 4/8/12-view는 분포 밖 사용이다. 실제로 off(=MVSplat 단독) 평균 PSNR이 2-view 25.4dB → 8-view 19.0dB로 view가 늘수록 오히려 나빠진다 — 분포 밖이라 MVSplat이 혼란스러워하는 것. 즉 "12-view에서 +10.5dB 개선"은 "view가 많을수록 refinement가 좋다"보다는 **"MVSplat이 분포 밖에서 만든 나쁜 초기값을 refinement가 복구·역전시킨 효과"**로 해석하는 게 더 정확하다. 이 교란요인을 분리하려면 2~6-view가 전부 분포 안인 DepthSplat으로 같은 실험을 반복해야 한다 — 내일 1순위로 격상.

**DL3DV overlap 이식**: RE10K와 같은 COLMAP 기반 파이프라인을 DL3DV pilot 25 scene에 이식(`core/dl3dv_dataset.py`, `analysis/generate_dl3dv_view_overlap.py`). 카메라 변환은 DepthSplat 공식 변환 스크립트로 재현. 결과: 2-view는 25개 전부 overlap 0(세 번째 데이터셋에서도 재현), 그런데 **4-view도 14/25(56%)가 overlap 0**으로 RE10K보다 훨씬 나쁘게 나왔다. 원인을 파보니 RE10K는 MVSplat 공식 context/target index(근접 프레임 제약이 이미 반영됨)를 그대로 썼는데 DL3DV는 그런 공식 index가 없어서 전체 영상에서 순수 랜덤으로 뽑았기 때문 — DepthSplat 실제 알고리즘(`view_sampler_bounded_v2.py`)을 코드로 확인해보니 test 시점엔 영상 시작부터 고정 window 안에서 farthest-point 샘플링을 쓰지, 전체 영상 랜덤이 아니었다. 이 결과는 "DL3DV 자체 성질"이 아니라 "우리 선택 방식 탓"일 가능성이 커서, 공정한 비교로 쓰려면 재작업이 필요하다고 명시해뒀다.

## 10. JᵀJ-overlap confound 재해석 + DepthSplat C1-b 연결 + DL3DV view 선택 수정 (밤 세션)

**JᵀJ-overlap confound, A-1 완료 후 재해석**: 사용자가 가우스-뉴턴 학습을 끝내고 직접 분석에 참여. `corr(overlap, log(uncertainty))=+0.952`(raw, H1 기대와 반대 부호)를 baseline으로 선형 통제하니 +0.801까지만 줄어서 "다른 원인이 더 있나" 하고 `shared_points`(평균에 들어가는 점 개수) 가설을 세워 검증했으나 기각(partial corr +0.071, 무관). 알고 보니 baseline-uncertainty 관계가 선형이 아니라 거듭제곱(log-log corr -0.989)이라, 선형 통제가 애초에 잘못된 도구였음을 발견 — log(baseline)으로 다시 통제하니 +0.301로 크게 줄었다. **결론: 원래 가설(baseline 교란)이 맞았고, 첫 선형 통제의 함수형 오지정이 가짜 잔차를 만들었던 것.** 전체 유도·해석·논문 초안 문장은 `paper/paper_geometry_confound_analysis_2026-08-12.md`에, 재현 코드는 `geometry_uncertainty_figure.py::print_confound_analysis()`에 통합.

**DepthSplat을 C1-b에 연결**: `runners/depthsplat_dl3dv_runner.py` 신규(MVSplat 러너와 같은 gaussians.pt/render_reference.pt 포맷). 로컬 체크포인트가 DL3DV 전용이라 DL3DV로 진행. 1 scene 검증: 렌더 등가성 gate 43~54dB(MVSplat의 35~42dB보다 정밀), warm-start baseline이 DepthSplat 자체 평가와 0.02dB 이내로 일치.

**DepthSplat C1-b 25-scene 스케일업 — 오늘의 핵심 결과**:

| 모델/데이터셋 | 2-view delta | 4-view delta | off 평균(2→4-view) |
|---|---:|---:|---|
| MVSplat/RE10K(4-view는 분포 밖) | -0.14dB | **+3.67dB** | 25.4→**20.2dB**(나빠짐) |
| DepthSplat/DL3DV(둘 다 분포 안) | +0.10dB | +0.15dB | 10.9→**17.0dB**(좋아짐) |

MVSplat은 view가 늘수록 refinement 효과가 폭발적으로 커지는데(-0.14→+3.67) DepthSplat(분포 안)은 거의 그대로다(+0.10→+0.15). off(FF 단독) 점수 자체도 MVSplat은 view 늘수록 나빠지고 DepthSplat은 좋아진다. **오전에 나온 "view 수 효과"가 사실은 "MVSplat이 분포 밖에서 무너지는 걸 refinement가 복구하는 효과"였다는 가설이 확인됐다.** 단, 모델도 데이터셋도 다르고 DL3DV view 선택은 v1(수정 전)이라 완전히 깨끗한 비교는 아니다.

**DL3DV view 선택 수정**: DepthSplat 실제 test-time 알고리즘(`view_sampler_bounded_v2.py`: window=[0,50] 고정 + farthest-point sampling, 전체 영상 랜덤 아님)을 그대로 재현하도록 `generate_dl3dv_view_overlap.py`를 재작성. 25 scene 재실행 결과 4-view zero-overlap 비율이 **56%(14/25) → 4%(1/25)**로 급감, median overlap 0.000→0.615 — 오늘 오후에 세웠던 가설(우리 선택 방식 탓)이 맞았다. 결과는 `dl3dv_overlap_v2/`(기존 `dl3dv_overlap/`은 동시에 돌던 DepthSplat 스케일업이 읽고 있어서 보존).

---

## 종합 결론

1. §5.2/§5.4가 채워지면서 연구설계/프로토콜 카테고리가 사실상 완료됐다. 남은 미결은 §5.11/§5.12 동결뿐(C2 budget은 오늘 저녁 확정).
2. densification on/off는 코드만이 아니라 실측 궤적으로 검증됐다 — C1-b 실험이 실제로 돌아갈 준비가 됐다.
3. RE10K, DL3DV 모두에서 "2-view는 SfM이 완전히 죽는다"가 재현됐다 — DTU까지 포함해 세 데이터셋 전부에서 나온, sparse-view 자체의 구조적 성질이라는 증거. 논문 핵심 서사(H1)에 바로 쓸 수 있는 결과다.
4. **V3(C1-b) 파이프라인이 DTU+RE10K+DL3DV, MVSplat+DepthSplat 두 모델 모두에서 완성·검증됐다.** RE10K/MVSplat에서 나온 "view 수가 늘수록 refinement 효과가 커진다"는 패턴이, DepthSplat(분포 안 모델)으로 반복해보니 사라졌다 — MVSplat의 분포 밖 사용이 진짜 원인이었다는 게 오늘 밤 확인됐다. 교란요인 하나를 완전히 풀어낸 것.
5. A-1(가우스-뉴턴) 완료 후 사용자와 직접 분석한 결과, geometry uncertainty의 overlap-baseline confound도 "가짜 잔차"였음이 밝혀지고 정리됐다 — 논문에 바로 쓸 수 있는 문장까지 작성됨.
6. DL3DV view 선택 방법론 문제도 오늘 안에 발견하고 수정까지 마쳤다 — 발견-가설-검증-수정이 하루 안에 한 바퀴 돌았다.

---

## 다음 실행 목록 (8/13로 이월)

1. DepthSplat C1-b를 DL3DV v2(고친 view 선택)로 재실행 — 더 깨끗한 MVSplat-vs-DepthSplat 비교 확보
2. Vanilla3DGS/MVSplat "일반"(non-warm-start) 경로를 RE10K main subset 256×256 입력으로 실제 실행 — COLMAP/random init 연결 필요, C1-a(진짜 Regime Map)로 가는 마지막 관문
3. renderer_equivalence_tolerance 최종 동결 — 여러 scene 실측 반영
4. DepthSplat 정식 승격, co-visibility selector 연결
5. SparseGS 통합 (0%, 아직 미착수)
6. RE10K citation/license 문구, train/ split 39 scene 문서 정합성(2026-08-12 오전에 이미 수정함, 완료)
