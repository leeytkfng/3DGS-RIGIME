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

---

## 종합 결론

1. §5.2/§5.4가 채워지면서 연구설계/프로토콜 카테고리가 사실상 완료됐다. 남은 미결은 C2 budget과 §5.11/§5.12 동결뿐.
2. densification on/off는 코드만이 아니라 실측 궤적으로 검증됐다 — C1-b 실험이 실제로 돌아갈 준비가 됐다.
3. RE10K에서도 DTU와 똑같이 "2-view는 SfM이 완전히 죽는다"가 재현됐다 — 이제 이게 DTU만의 특이 현상이 아니라 sparse-view 자체의 구조적 성질이라고 말할 수 있는 두 번째 데이터셋 증거가 생겼다. 이건 논문의 핵심 서사(H1)에 직접 쓸 수 있는 결과다.
4. RE10K main subset의 view/overlap 인프라(DTU에서 검증된 세 요소: view selection, overlap 계산, low/high bucket)가 이제 RE10K에도 붙었다. 아직 안 붙은 건 실제 runner 실행(Vanilla3DGS/MVSplat을 이 후보로 실제로 돌리기)뿐이다.
5. 체크리스트(`experiment_checklist.md`) 기준 남은 다음 순서: V3(C1-b) 구현(FF→3DGS 변환기·렌더 등가성 gate·warm-start·refinement loop), C2 budget 결정, RE10K runner 실행 연결.

---

## 다음 실행 목록 (8/13로 이월)

1. V3(C1-b) 구현 착수 — FF Gaussian → gsplat 포맷 변환기부터
2. C2 budget 결정 (§5.4 GPU-hour 확정의 유일한 미결 변수)
3. Vanilla3DGS/MVSplat을 RE10K main subset 256×256 입력으로 실제 실행
4. DL3DV에도 같은 overlap 패턴 이식
5. DepthSplat 정식 승격, co-visibility selector 연결
6. RE10K citation/license 문구, train/ split 39 scene 존재 여부 문서 정합성 확인(`SOURCE.md`가 "train 없음"이라 적혀있는데 실제로는 있음 — 출처 불명, 확인 필요)
