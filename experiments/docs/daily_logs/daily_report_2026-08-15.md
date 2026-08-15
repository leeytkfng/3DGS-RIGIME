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

## 6. main.tex §4 Results 채움 + 팀 역할 재확인

**실험 목적**: RE10K C1-a가 끝났으니 논문 초안의 Results 절(지금까지 `\todo`/`\chk` placeholder였던 부분)을 실제 수치로 채운다.

**데이터/특징**: `regime_map.png`/`pareto_frontier.png`를 표~\ref{tab:regime}·그림 자리에 실제 삽입. 승패 판정 표는 세 방법 쌍(Vanilla3DGS-vs-MVSplat, FSGS-vs-MVSplat, FSGS-vs-Vanilla3DGS) 전부를 view 수 × budget 격자로 채웠다. "장면별 win rate와 신뢰구간" 절에는 CI까지 곁들여 두 지점을 서술했다 — FSGS가 8-view·budget≥60s에서 MVSplat을 역전한 것(30-scene 완비 후 robust하게 유지됨)과, Vanilla3DGS가 12-view에서 예산에 대해 비단조(10s에 앞서다가 60s에 떨어지고 300s에 회복)라는 아직 원인 불명인 패턴.

**팀 역할 재확인**: 사용자가 `팀 실행 가이드.pdf`(이용수/황인재/서창희 3인 역할표)를 공유해 확인한 결과, **C1-b 실행은 황인재(2저자) 담당**임이 명확해졌다(이용수는 C1-b의 전제조건인 렌더 등가성 gate만 담당, 이미 완료). 이 구조를 memory에 기록해뒀다 — 앞으로 C1-b 실행에는 먼저 손대지 않고 확인만 하기로.

**논문 연결**: commit `c3c81a2`.

---

## 7. DL3DV 대시보드 생성 + 시간당 자동 갱신

**실험 목적**: RE10K 대시보드처럼 DL3DV 진행 상황도 실시간으로 볼 수 있게 요청받음. 추가로 "1시간마다 자동 갱신"도 요청받아 자동화 방법을 검토.

**막힌 점과 해결**: 처음엔 클라우드 스케줄(`/schedule`, RemoteTrigger 기반 cron routine)로 시도했으나, 클라우드 루틴은 완전히 격리된 샌드박스에서 돌아 이 로컬 머신의 실험 출력 파일(`experiments/outputs/dl3dv_c1a_main/c1a_main_summary.json`)에 접근할 수 없다는 걸 확인 — 만들어도 빈 데이터로 덮어쓰기만 할 뿐이라 무의미해서 중단. 대신 **이 세션 안에서 도는 cron**(`CronCreate`, 매시 07분, 세션 유지되는 동안 · 최대 7일)으로 전환.

**데이터/특징**: RE10K 대시보드 HTML을 템플릿으로 삼아 DL3DV용으로 복제·수정(제목, 방법명 MVSplat→DepthSplat, TOTAL_COMBOS 30→25, 범례·footer 텍스트). 새 Artifact로 발행: `https://claude.ai/code/artifact/24c35b05-4da3-4877-a37b-bf38811fdbf5`.

**결과**: cron이 매시 07분에 (1) 진행률 확인 (2) `generate_regime_map.py`로 그림 재생성 (3) 대시보드 ROWS/SNAPSHOT_AT 갱신 후 재배포를 자동 수행. 첫 실행에서 `generate_regime_map.py`가 방법명(FF)을 `MVSplat`으로 하드코딩해뒀던 걸 발견해 `--ff-method`/`--dataset-label` CLI 옵션으로 일반화했고, scene 1개뿐인 초반 구간에서 Pareto 그림이 음수 오차막대로 에러나던 버그도 같이 고침.

**논문 연결**: 없음(운영 인프라). commit `37c4a88`.

---

## 8. H2 가설 검증 — RE10K 궤적 로그 분석

**실험 목적**: "내가 맡은 최종 목표가 뭐냐"는 질문에 답하며 정리한 우선순위(§C2, H가설) 중 H2("view 수가 적을수록 optimization 품질 정점이 빨리 오고 하강이 가파르다")는 이미 완료된 RE10K C1-a 궤적 로그만으로 바로 검증 가능하다는 걸 확인, 즉시 착수.

**데이터/특징**: `h2_dynamics_analysis.py` 신규 작성. Vanilla3DGS·FSGS의 (scene, view_count, seed) 궤적(예산 스냅샷 4개: 1/10/60/300s) 480개에서, 4개 중 test PSNR 최댓값이 300s 이전에 나오는 비율("조기 정점 비율")과 조기 정점 시 정점 대비 300s 하강폭을 view 수별로 집계(하강폭은 scene cluster bootstrap CI).

**결과**: 가설이 정확히 절반만 맞았다. **정점 시점**은 지지됨 — 조기 정점 비율이 view 수가 늘수록 단조 감소(Vanilla3DGS 91.7%→63.3%, FSGS 91.7%→18.3%, 2→12-view). **하강 기울기**는 기각(그것도 정반대)됨 — Vanilla3DGS는 view 수가 늘수록 하강폭이 오히려 커짐(2-view 0.69dB → 12-view 2.80dB), FSGS는 뚜렷한 추세 없음. 12-view Vanilla3DGS의 이 큰 하강폭은 §7에서 서술한 "12-view 예산 비단조성"과 같은 원인일 가능성이 있어 보인다.

**논문 연결**: `main.tex` 가설 표 H2 행 갱신 + 새 문단 추가. commit `ea97e0a`.

---

## 9. C2 설계 착수 — depth back-projection 모델 확보

**실험 목적**: H2 다음으로 C2(depth noise 개입 실험) 설계 착수. 시작하자마자 막힌 지점부터 확인.

**막힌 점**: overall.md §5.9는 C2의 depth 개입 트랙을 "VGGT/DA3 계열 depth back-projection 초기화"로 명시하는데, 확인해보니 이 환경 어디에도(모든 conda env, `/data/Re-feem`) 그런 모델이 설치돼 있지 않았다 — C2는 코드가 전혀 없는 상태에서 시작해야 했다.

**결정**: VGGT 대신 **Depth Anything V2 Metric**을 쓰기로 했다. 우리 트랙은 pose-given(카메라 pose를 이미 알고 있음)이라 VGGT의 pose 추정 기능이 불필요하고, VGGT는 자체 좌표계로 point map을 출력해 우리 world 좌표계로 재정렬해야 하는 반면 DepthAnything-V2-Metric은 known intrinsics로 바로 back-projection 가능한 metric depth를 직접 준다 — 파이프라인이 더 단순하다.

**데이터/특징**: 전용 conda env `depth` 신규 생성(torch/torchvision/transformers/pillow만 설치, 기존 env와 버전 충돌 방지). `core/depth_model_smoke_test.py` 작성해 DTU scan1 이미지 1장으로 실제 추론 확인.

**결과**: 정상 동작 확인 — depth 범위 0.537~3.006m(DTU가 근접 촬영 tabletop object라 이 정도가 정상 범위), NaN/Inf 없음, 추론 0.33초. 모델 설치·검증까지는 끝났고, 아직 남은 건 (1) `vanilla_3dgs_runner.py`에 `--init-source depth_backprojection` 경로 구현(noise/scale-bias 주입 포함), (2) DTU에도 RE10K/DL3DV처럼 co-visibility selector를 적용해 `representative_conditions`(2view_low_overlap 등 4개)에 실제 DTU scene 배정. GPU는 지금 DL3DV가 쓰고 있어 실제 300s 실행은 DL3DV 완료 후로 순연.

**논문 연결**: `overall.md` §5.9에 2026-08-15 항목으로 결정 근거 기록.

---

## 오늘 할 일

1. ~~RE10K 완료 대기~~ — **완료됨** (§5).
2. **DL3DV 본 실험** — 진행 중, cron으로 매시 자동 모니터링 (§7).
3. **C2 depth-init 러너 구현** — `vanilla_3dgs_runner.py`에 depth back-projection + noise/scale-bias 주입 경로 추가 (§9 후속).
4. **DTU co-visibility selector 적용** — C2 representative_conditions 4개에 실제 scene 배정 (§9 후속).
5. **정성적 렌더 비교 PNG** (GT/MVSplat/Vanilla3DGS/FSGS) — 이미 완료된 체크포인트에서 생성 착수 예정, 아직 미착수.
6. **DepthSplat C1-b v2 재실행** — **황인재 담당으로 확인됨**(§6), 우리 쪽에서 먼저 손대지 않는다.
7. **RE10K/DL3DV 베이스라인-overlap confound 확인** — DTU(궤도 rig)에서 나온 관계가 path-trajectory 데이터셋에서도 같은지, 아직 미착수.
8. **ReSplat 스코프 결정 대기** — 사용자가 원문 다 읽으면 spot-check 방식(2-view RE10K, 8-view DL3DV) 실행 여부 확정.
