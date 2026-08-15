# 일일 보고서 — 2026-08-15

## 오늘 목표

RE10K C1-a 본 실험(30-scene) 마무리 대기. 사용자가 ReSplat/Diff3R/SplatFormer를 직접 읽는 동안, 병행 가능한 작업(그림 생성, 대시보드 유지, ReSplat 스코프 조사, DL3DV 착수 준비)을 진행한다.

---

## 1. 대시보드 raw-cell-data 표의 n 표시 버그 발견 및 수정

**실험 목적**: 사용자가 대시보드에서 방법(method)별 `n` 값이 26 / 25 / 50 식으로 서로 다르게 찍힌 걸 보고 원인을 물어봐서 확인.

**데이터/특징**: `c1a_dashboard.html`의 `aggregate()` 함수(JS)가 (view_count, budget, method)로만 묶고, 그 안의 test_psnr row를 scene 구분 없이 그냥 다 세고 있었다.

**쉽게**: MVSplat은 seed가 없어서 scene당 결과가 1개인데, Vanilla3DGS·FSGS는 seed 0·1을 각각 따로 재구성해서 scene당 결과가 2개다. 그래서 다 같은 진행 상황이어도 화면에는 "MVSplat은 표본 26개, FSGS는 25개, Vanilla3DGS는 50개"처럼 서로 다른 숫자로 보였다 — 실제로는 각 방법이 커버한 scene 수가 비슷한데, seed 두 번 돈 방법만 숫자가 두 배로 부풀려진 것.

**중요한 점**: 이건 화면 표시 버그일 뿐, 실제 통계 계산(`core/protocol_utils.py::scene_cluster_bootstrap_ci`, `generate_regime_map.py`)은 원래부터 seed를 scene 안에서 먼저 평균 낸 뒤 scene 수를 독립 표본 크기로 쓰고 있어서 논문에 들어갈 실제 결과(τ 판정, regime map, Pareto)에는 영향이 없었다.

**수정**: `aggregate()`를 scene별로 먼저 grouping해서 seed 평균을 낸 다음, distinct scene 수를 `n`으로 세도록 변경. 테이블 헤더도 `n (scenes)`로 명시. 최신 데이터(102/120 combo)로 재배포 완료: [대시보드](https://claude.ai/code/artifact/ff507d8b-56f7-448a-a547-c378e7f25676)

**논문 연결**: 없음(논문 결과 자체는 처음부터 정상). 다만 사용자가 raw 숫자만 보고 "표본 크기가 방법마다 다르다"고 잘못 해석할 뻔했던 걸 미리 잡은 셈 — 앞으로 대시보드에 새 집계 열을 추가할 때는 scene을 독립 단위로 묶는 원칙을 계속 지켜야 한다.

---

## 2. Regime map / Pareto frontier 그림 갱신 스크립트 완성 및 실행

**실험 목적**: 논문 Figure 우선순위 ①②(Regime Map, 품질-시간 Pareto)를 지금 있는 부분 데이터로 미리 만들어두기 위해.

**데이터/특징**: `generate_regime_map.py` 신규 작성. `c1a_main_summary.json`을 읽어 방법 쌍(Vanilla3DGS-vs-MVSplat, FSGS-vs-MVSplat, FSGS-vs-Vanilla3DGS)별로 view_count × budget 승패 지도를, `scene_cluster_bootstrap_ci` + view_count-tiered τ(0.5dB@2/4-view, 1.4dB@8/12-view)로 계산해 3-class(A승/B승/Tie) 시각화. Pareto frontier는 view_count별 품질-시간 곡선(2×2 facet).

**결과**: 90/120 combo 시점 기준 방향성 확인 — FSGS가 4-view부터 Vanilla3DGS를 역전, 8-view부터는 전 구간 우세. FSGS vs MVSplat은 8-view·60s에서 근소하게(Δ=1.44dB, τ=1.4dB) 역전됐지만 신뢰구간이 크게 겹쳐 아직 불확실(scene 22/30개만 반영), 300s에서는 다시 Tie로 돌아옴 — 30-scene 완료 전까지는 노이즈 가능성으로 보류.

**논문 연결**: `experiments/docs/paper/overleaf_draft/figures/regime_map.png`, `pareto_frontier.png` 커밋 완료(commit `53fa7cd`). RE10K 완료 후 최종판으로 한 번 더 갱신 예정.

---

## 3. DL3DV 본 실험 오케스트레이터 작성

**실험 목적**: RE10K 끝나는 즉시 DL3DV로 넘어가기로 한 결정(2026-08-14)에 따라 코드를 미리 준비.

**데이터/특징**: `run_dl3dv_c1a_main.py` 신규 작성 — `run_re10k_c1a_main.py` 구조를 그대로 이식하되 feed-forward만 MVSplat→DepthSplat으로 교체. DL3DV 25-scene, view_count [2,4,8,12], budget [1,10,60,300]s, seed{0,1}. overlap_level 축은 이번 스코프에서 제외(RE10K와 동일 원칙).

**결과**: 문법 검증(`ast.parse`) 통과. RE10K 작업 중이라 실행은 보류, 코드만 대기 상태. commit `53fa7cd`.

**논문 연결**: RE10K가 끝나는 대로(오늘 UTC 11시경 예상) 바로 착수.

---

## 4. ReSplat 4번째 방법 포함 여부 — 체크포인트 조사

**실험 목적**: 사용자가 ReSplat을 읽으면서 "순환 최적화라 결이 하나 늘어난다"며 실험 포함 여부를 고민 중이라, 판단에 필요한 사실관계(체크포인트 공개 여부)를 먼저 조사.

**데이터/특징**: 공식 저장소 `cvg/resplat`(ECCV'26 Oral, MIT license) Model Zoo 확인. RE10K 체크포인트는 **2-view, 256×256** 하나만 공개(우리 MVSplat 설정과 정확히 일치). DL3DV는 8/16/32-view 체크포인트가 있고 그중 **8-view, 256×448**이 우리 DepthSplat 설정과 일치. 4-view·12-view용 공개 체크포인트는 없음.

**결론(잠정)**: 풀그리드 4번째 방법으로는 불가능(체크포인트 부족), 대신 "RE10K 2-view + DL3DV 8-view 두 지점만 spot-check"는 체크포인트가 있어 엔지니어링 비용이 낮음 — 기존 FF 러너 구조 재사용 가능. 최종 결정은 사용자가 원문 다 읽은 뒤.

**논문 연결**: `main.tex`의 "경계 지대를 활용하는 연구" 문단에서 이미 ReSplat을 소개해뒀음 — 포함하면 그 서술을 실측 데이터로 뒷받침하는 그림이 됨.

---

## 5. RE10K C1-a 본 실험 완료 (30-scene, 120 combo, 2400 row)

**실험 목적**: 2026-08-13 밤에 착수한 RE10K C1-a 본 실험(MVSplat vs Vanilla3DGS vs FSGS, view_count [2,4,8,12] × budget [1,10,60,300]s × seed{0,1})이 완료됐는지 확인.

**데이터/특징**: `run_re10k_c1a_main.py` 프로세스가 14:08 UTC경 종료(30 scene 전체 처리 완료). `c1a_main_summary.json` 검사 결과 120/120 combo는 채워졌지만, **딱 두 개의 개별 하위 실행이 조용히 실패**해 2400개 기대 row 중 2388개만 있었다 — scene `a9b3ff60b213e099`의 (view_count=8, Vanilla3DGS, seed 0·1 둘 다)와 (view_count=12, MVSplat)이 빠져있었다.

**쉽게**: 원인을 재현해보니 3DGS 렌더러(gsplat)가 CUDA 확장을 JIT 컴파일할 때 필요한 `ninja` 빌드 도구를 못 찾아서 실패한 것. 오케스트레이터 원본 코드는 `_env_with_bin()`으로 conda 환경의 `bin` 경로를 PATH에 미리 넣어줘서 평소엔 문제가 없는데, 이 두 실행에서만 그게 왜 실패했는지는 정확히 못 밝혔다(캐시된 빌드가 이미 있었는데도 재빌드를 시도한 정황은 확인 — 아마 이 시점에 다른 프로세스와의 일시적 경쟁 상태였을 가능성). 오케스트레이터는 실패해도 멈추지 않고 다음 단계로 계속 진행하도록 짜여 있어서(의도된 동작 — 크래시 시 전체 job이 죽는 것보다 안전), 이 두 구멍만 남기고 끝까지 완주했다.

**수정**: 두 실행을 conda env `bin` 경로를 PATH에 명시적으로 포함해 개별 재실행 → 둘 다 정상 완료. `c1a_main_summary.json`에 누락 row 12개(MVSplat 4개 + Vanilla3DGS 8개)를 직접 append해 병합.

**결과**: 최종 **120/120 combo, 30 scene, 2400/2400 row, 전부 status="ok"**. 결측치 없는 완전한 데이터셋 확보.

**논문 연결**: `regime_map.png`, `pareto_frontier.png`를 최종 완전 데이터로 재생성, 대시보드도 "complete" 스냅샷으로 재배포. 이제 이 데이터가 C1-a RE10K 최종 결과다 — 통계 분석(τ 판정, Holm 보정)을 이 기준으로 확정해도 된다.

---

## 오늘 할 일

1. ~~RE10K 완료 대기~~ — **완료됨** (§5).
2. **DL3DV 본 실험 착수** — `run_dl3dv_c1a_main.py` 실행, RE10K와 동일 규모(25 scene × 4 view_count × budget × seed).
3. **정성적 렌더 비교 PNG** (GT/MVSplat/Vanilla3DGS/FSGS) — 이미 완료된 체크포인트에서 생성 착수 예정, 아직 미착수.
4. **DepthSplat C1-b v2 재실행** (`dl3dv_overlap_v2` 기준) — 체크리스트 미해결 항목, DL3DV 본 실험과 GPU 일정 조율 필요.
5. **RE10K/DL3DV 베이스라인-overlap confound 확인** — DTU(궤도 rig)에서 나온 관계가 path-trajectory 데이터셋에서도 같은지, 아직 미착수.
6. **ReSplat 스코프 결정 대기** — 사용자가 원문 다 읽으면 spot-check 방식(2-view RE10K, 8-view DL3DV) 실행 여부 확정.
