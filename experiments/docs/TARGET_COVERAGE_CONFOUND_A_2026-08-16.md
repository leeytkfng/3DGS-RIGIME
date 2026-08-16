# Target-Coverage Confound 진단 — A-2 대응 (2026-08-16)

**작성**: 이용수(A 컨테이너) — 작업 지시 A-2 대응
**성격**: 탐색적 진단. **여기서는 진단만 하고 `core/view_selector.py` 변경은 하지 않는다.** 결과를
보고 selector를 바꾸는 것은 프로토콜 변경이므로 공동 승인이 필요하다(A-2 지시문 원문 그대로).

**스크립트**: `experiments/scripts/analysis/target_coverage_confound.py`
**원자료**: `experiments/outputs/target_coverage_confound/rows.json` (440 rows = RE10K 30scene×4viewcount×2level + DL3DV 25scene×4viewcount×2level)
**입력**: `re10k_overlap_candidates.json`(30 scene), `dl3dv_overlap_lowhigh/dl3dv_overlap_candidates.json`(25 scene) — 이미 생성돼 있던 파일 그대로 사용, 이번 세션에 새로 만들지 않음.

---

## 1. 질문

`core/view_selector.py`의 high overlap 후보(좁은 랜덤 window 안 FPS)와 low overlap 후보(전체
범위 FPS)가, overlap 축과 무관하게 "target을 얼마나 잘 둘러싸는가/얼마나 가까운가"도 같이 바꾸는지.
바뀐다면 향후 이 selector로 측정할 overlap 축 결과가 이 배치 효과와 뒤섞일 수 있다.

## 2. 방법

각 (scene, view_count, level) 조건마다 target view(들)에 대해 세 지표를 계산하고, target이
여러 개면 평균한다:

- **bracket_rate**: target camera center가 context camera center들의 convex hull 안에 있는지
  (boolean). Context가 3개 미만이면 hull 자체가 불가능해 "해당 없음"으로 제외(2-view 조건).
  3D hull이 축퇴되면(거의 공면) 주성분 2D 평면에 투영 후 재판정.
- **min_angle_deg**: target view의 카메라 forward 방향과, 가장 가까운 context view forward
  방향 사이의 최소 각도.
- **centroid_dist / nearest_dist**: target center와 context centroid/최근접 context center
  사이 거리.

Scene을 독립 단위로 하는 `scene_cluster_bootstrap_ci`(기존 프로토콜과 동일 방식)로 high/low
각각의 평균·CI를 내고, scene별 (high−low) paired delta도 같은 방식으로 CI를 냈다.

## 3. 결과

### 3.1 Bracketing — 이 지표는 이 데이터셋들에 잘 안 맞는다

RE10K/DL3DV 둘 다 단일 연속 궤적을 도는 forward-facing 촬영이라, context가 target을 3D로
완전히 "둘러싸는" 경우 자체가 구조적으로 드물다. bracket_rate가 모든 조건에서 0~20% 수준이고,
high/low 사이 paired delta의 CI가 대부분 0을 포함하거나(±0.1 남짓의 작은 차이) 부호가 오히려
역전돼 있다(8/12-view에서 low가 살짝 더 높음, 하지만 CI가 넓어 확정적이지 않음). **A-2가 제안한
"넓게 퍼진 low가 target을 더 잘 감싼다" 가설은 이 두 데이터셋(둘 다 단일 궤적 촬영)에서는 뚜렷하게
관측되지 않는다** — 애초에 bracketing이 거의 안 일어나는 기하이기 때문.

### 3.2 각도/거리 — 훨씬 강하고 일관된 다른 확인 결과

Bracketing 대신, **min_angle_deg와 nearest_dist에서 매우 일관되고 통계적으로도 뚜렷한 차이**가
나왔다. view_count≥4 모든 조건에서 **low overlap 후보가 high overlap 후보보다 target에 각도상
더 가깝고(정렬이 더 잘 맞고) 물리적으로도 더 가깝다**:

| Dataset | view_count | Δ min_angle_deg (high−low) | Δ nearest_dist (high−low) |
|---|---|---|---|
| RE10K | 4 | +3.32° [+1.67,+5.38] | +0.336 [+0.164,+0.616] |
| RE10K | 8 | +2.79° [+1.13,+4.93] | +0.307 [+0.128,+0.587] |
| RE10K | 12 | +1.84° [+0.41,+3.96] | +0.215 [+0.047,+0.489] |
| DL3DV | 2 | +25.17° [+13.44,+37.52] | +1.690 [+0.989,+2.327] |
| DL3DV | 4 | +30.83° [+20.98,+41.78] | +1.888 [+1.238,+2.474] |
| DL3DV | 8 | +31.46° [+21.92,+42.04] | +1.877 [+1.360,+2.348] |
| DL3DV | 12 | +27.77° [+17.75,+38.48] | +1.819 [+1.383,+2.230] |

(양수 delta = high가 low보다 각도/거리가 더 크다 = low가 target에 더 가깝다/더 잘 정렬된다.)

DL3DV는 모든 view_count에서, RE10K는 4/8/12-view에서 paired CI가 0을 완전히 벗어난다(2-view
RE10K만 CI가 0을 살짝 포함해 약한 신호). DL3DV는 delta 크기 자체도 RE10K보다 훨씬 커서(20~30°대)
효과가 더 뚜렷하다.

**메커니즘**: `select_high_overlap_indices`는 전체 pool 중 무작위로 배치된 좁은 window
(`window_start ~ Uniform`) 안에서만 FPS를 돌린다 — 이 window가 target이 위치한 궤적 구간
근처에 걸릴지는 순전히 우연이다. 반대로 `select_low_overlap_indices`는 pool 전체에서 FPS를
돌리는데, pool 자체가 "target을 제외한 나머지 전부"라 target 바로 인접 구간도 pool에 포함돼
있다. 전체 범위를 고르게 커버하려는 FPS의 특성상, 그 인접 구간에서도 최소 한두 개는 뽑히는
경우가 많다 — 결과적으로 "넓게 퍼뜨렸을 뿐"인 low 후보가 오히려 target 근처의 view를 하나쯤
포함할 확률이, "무작위 위치의 좁은 창"인 high 후보보다 구조적으로 높다.

## 4. 해석 — 어느 방향으로 작용하는 confound인가

이 결과가 뜻하는 바는 A-2가 원래 제안한 방향과는 다르지만(그 대신 bracketing이 아니라
각도/거리 정렬), 여전히 실질적인 confound다: **만약 이 selector로 만든 high/low 조건을 갖고
overlap 축의 효과를 측정한다면, "low overlap" 쪽 결과에는 실제 overlap 감소 효과뿐 아니라
"target에 더 가깝고 더 잘 정렬된 context를 우연히 더 많이 포함"하는 효과가 섞여 들어간다.**
방향상 이건 low overlap 조건에 유리하게(즉 overlap이 낮을수록 손해가 실제보다 작아 보이게)
작용할 가능성이 있다 — target 근접/정렬은 일반적으로 novel-view 품질에 유리한 조건이기 때문.

## 5. 제한 사항

- 2-view 조건은 bracketing 계산이 원천적으로 불가능(context 2개로는 hull을 못 만듦) —
  "해당 없음"으로 표시, 0이 아님에 유의.
- 이 selector(`view_selector.py`)는 **이번 세션에 새로 만든 모듈**로, 지금까지 실측된 C1-a
  본 실험(RE10K/DL3DV 2400+2000행)에는 쓰이지 않았다 — 즉 이 confound는 기존 결과를 오염시키지
  않았다. 앞으로 이 selector를 실제 overlap 축 실험에 쓸 계획이 있을 때만 해당하는 진단이다.
- forward-facing 단일 궤적 촬영이라는 두 데이터셋의 공통 특성 때문에 나온 결과라, 다른 종류의
  캡처(예: 360도 surround)에는 그대로 일반화되지 않을 수 있다.

## 6. 제안 (변경 아님 — 검토용)

**아래는 제안일 뿐, 이 세션에서 실행하지 않았다. 실행하려면 공동 승인 필요.**

1. 이 selector를 실제 overlap 축 실험에 쓸 경우, 결과 보고 시 "high/low 조건의 target 각도/거리
   정렬 차이가 있었다"는 제한 문구를 함께 명시.
2. 완화 방안 후보: `select_high_overlap_indices`의 window 배치를 무작위 대신 target 인접
   구간을 반드시 포함/제외하도록 고정하는 방법 — 다만 이 경우 low와 high 후보의 "차이가
   overlap만"이라는 selector 설계 원래 취지(모듈 docstring 참고)가 깨질 수 있어 신중한 검토
   필요.
3. 최소 조치로는, 향후 이 selector로 실험을 돌릴 때 이 스크립트로 사전 진단해서 target 정렬
   차이가 유의한 view_count 구간만이라도 인지하고 해석에 반영.

## 7. 재현

```
PATH="/opt/conda/envs/ps3/bin:$PATH" python3 experiments/scripts/analysis/target_coverage_confound.py
```
