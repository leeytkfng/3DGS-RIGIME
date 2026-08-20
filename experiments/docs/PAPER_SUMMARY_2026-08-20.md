# 논문 정리 — Sparse-view 3DGS Regime Study (2026-08-20)

원본: `experiments/docs/paper/overleaf_draft/`(Overleaf, KCI 2단 양식). 저자 이용수/황인재/서창희/최종선.

## 1. 연구 질문 & 핵심 메시지

새 알고리즘을 제안하는 논문이 아니라 **실증 연구(empirical study)**다.

> Sparse-view 3D 재구성에서 입력 view 수, view overlap, 계산 시간 예산에 따라 feed-forward와
> per-scene optimization의 품질--효율 우위는 어떻게 변화하며, 그 역전 경계는 무엇이 결정하는가?

핵심 메시지: **보편적으로 우월한 패러다임은 없으며, 실용적 선택은 view 수·overlap·예산·초기화 방식의 함수다.**

## 2. 논문 구조 (서사 사슬)

Regime Map(어디서 갈리는가) → 메커니즘 증거(왜 갈리는가) → 통제 검증(인과관계 확인) → 실용 가이드라인.

| 절 | 내용 |
|---|---|
| I 서론 | 연구 질문, 기존 문헌의 3가지 구조적 한계, 4개 기여 |
| II 관련 연구 | Per-scene optimization / Feed-forward / 경계 지대 연구(Diff3R, ForeSplat, ReSplat) / 기존 sparse-view 벤치마크와의 차이 |
| III 프로토콜 | 비교 대상, 데이터셋, 통제 축, overlap 정의, 승패 판정 규칙, 통계 분석 계획(scene cluster bootstrap), 조기 종료 규칙, 렌더 등가성 검증 |
| IV 결과 | Regime Map, Pareto frontier, refinement on/off 통제 실험, 초기화 비교, OOD 역전 지점 대조, ReSplat 탐색적 비교 |
| V 실패 분석 | baseline-overlap-불확실도 부호 역전, densification 관측 부족 메커니즘, H1/H2/H3 검증, C2 depth-noise sweep |
| VI 가이드라인 | 축별 실무 권장(View×예산 확정, overlap 실행 대기, depth-noise 강건성 확정) |
| VII 결론/한계 | 종합, 8개 한계 항목 |
| 부록 | A. 조기 종료(미착수), B. Oracle peak 분석 |

## 3. 방법론

- **비교 대상**: Feed-forward — MVSplat(RE10K, 2-view 학습), DepthSplat(DL3DV, 2–6-view 학습). Optimization — 3DGS(gsplat, COLMAP sparse init), FSGS(같은 sparse init으로 통일 — 원 논문의 dense-MVS init 대신, 공정성을 위한 의도적 선택).
- **데이터셋**: 주 RE10K 30-scene, 보조 DL3DV 25-scene pilot, 외부 검증 DTU 8-scan.
- **통제 축**: view 수 {2,4,8,12} × overlap {고/저, co-visibility 수치 정의} × 예산 {1,10,60,300s}.
- **승패 판정**: Δ=PSNR_FF−PSNR_OPT, τ=max(seed 변동성, 0.5dB) — 2/4-view는 0.5dB, 8/12-view는 1.4dB.
- **통계**: scene을 독립 단위로 한 scene cluster bootstrap(2000회 리샘플) 95% CI, 방법 쌍당 Holm–Bonferroni 보정.
- **학습 view 분포 이탈 confound**: feed-forward는 학습 분포를 벗어나면 성능이 저하되는데 optimization은 그 개념이 없음 — "정보량 가설" vs "분포 이탈 가설"을 분리하기 위해 (1) 학습 분포가 다른 두 FF 모델 대조, (2) 동일 초기값 refinement on/off 통제, (3) 서술 수위 제한("대표 시스템의 실용적 우위"로 한정) 세 갈래로 대응.

## 4. 핵심 결과

### 4.1 Regime Map (RE10K 30-scene, view×예산)
- 2/4-view: 모든 예산에서 feed-forward 우세.
- 8-view: 10s는 tie, 60s부터 FSGS 우세.
- 12-view: 10s부터 FSGS 우세. Vanilla3DGS는 예산에 비단조(10s 21.06dB → 60s 16.97dB → 300s 21.38dB) — 원인 미확정(개별 scene 불안정 vs 체계적 현상).

### 4.2 동일 초기값 Refinement On/Off (모델 고정, RE10K)
60s 시점 순효과(Δ = on−off): 2-view −0.14dB(CI가 0 포함, 사실상 무효과) → 4-view +3.67dB → 8-view +7.71dB → 12-view +10.47dB. View 수가 늘수록 refinement의 순수 기여가 단조 증가 — 다만 초기값(off) PSNR 자체가 view 수에 따라 나빠지는 것과 얽혀 있어 "정보량"과 "분포 이탈"이 여전히 완전히 분리되지는 않음.

### 4.3 초기화 비교: FF warm-start vs COLMAP sparse (동일 20-scene, 동일 최적화 절차)
60s 시점 gap(FF−COLMAP): 2-view +13.80dB, 4-view +14.88dB, 8-view +10.40dB, 12-view +11.18dB — **모든 view 수에서 10–15dB**, CI 전부 0에서 멀리 떨어짐. Scene별 분산이 매우 커서(−0.32~+29.10dB) "균일한 페널티"가 아니라 "60s 안에 densification이 장면을 다 커버하는지"가 scene마다 갈리는 문제로 해석. → **실용적 선택이 "FF 대 optimization" 이분법이 아니라 "COLMAP+opt 대 FF-init+opt"로 재구성될 수 있음을 시사** (predict-then-refine 계열의 설계 선택과 같은 방향).

### 4.4 OOD 역전 지점 대조 (MVSplat vs DepthSplat)
학습 view 분포가 더 넓은 DepthSplat(2–6-view) 쪽이 역전 지점도 더 바깥(8–12-view 사이)으로 밀림 — RE10K/MVSplat(고정 2-view 학습)은 8-view/60s부터 역전. 분포 이탈 가설을 부분적으로 뒷받침(완전 통제 비교는 아님 — 데이터셋 자체가 다름).

### 4.5 ReSplat 탐색적 비교 (사전 등록 밖, DL3DV)
Recurrent refinement 계열 ReSplat이 진짜 in-domain 지점(8-view)에서 4개 방법 중 최고(26.41dB, DepthSplat 대비 +0.6dB) — 그러나 12-view에서는 16-view 체크포인트를 써도 FSGS가 여전히 +2.2dB 우세. 핵심 서사(고-view에서 optimization 역전)가 더 강한 FF 기준선 아래서도 유지됨을 보여주는 참고 자료.

### 4.6 실패 분석 — 메커니즘
- **Baseline-overlap-불확실도 부호 역전**: DTU 궤도형 rig에서 원시 상관 corr(overlap, log 불확실도)=+0.952(예상과 반대 부호) — log(baseline) 통제 시 편상관 +0.301로 감소, 부호 역전의 주 원인이 baseline 교란임을 확인.
- **Densification 관측 부족**: "관측 적은 Gaussian이 densification 오탐을 유발한다"는 원 가설은 기각(count≤2인 Gaussian이 임계 초과하는 비율 0.00–0.003%). 대신 관측 부족 Gaussian **비율 자체**가 view 수에 강하게 좌우됨을 발견(2-view 51.5% → 12-view 0.7%, 초기화 고정 시에도 재현) — 탐색적 발견(사전 등록 가설 아님).
- **H1/H2/H3 가설 검증표**:

| 가설 | 내용 | 상태 |
|---|---|---|
| H1 | σ↑ → Gaussian 수↑, 품질↓, 저겹침에서 강화 | 절반 지지, 절반 기각 |
| H2 | view 적을수록 정점 시점 앞당겨지고 하강 기울기 가팔라짐 | 정점 시점만 지지, 하강 기울기는 정반대 |
| H3 | 초기 geometry 품질 충분하면 refinement 한계 이득 소멸 | 잠정 지지, 교란 미분리 |

### 4.7 C2 depth-noise sensitivity (DTU 8-scan, 640/640 combo 완주)
- σ(개별 노이즈)↑ → PSNR 하락(지지, 4개 조건 전부) 그러나 Gaussian 수는 **하락 또는 평탄**(H1 두 번째 절반 기각) — 노이즈가 gradient 신호 자체를 훼손해 densification이 억제되는 것으로 해석.
- s(전역 스케일 편향): 탐색 범위[0.9,1.1] 안에서 정점을 못 찾음(단조 증가) — 파이프라인 자체의 계통적 과소추정 편향 가능성.

### 4.8 부록 B — Oracle peak gap
예산 종료 시점(300s) vs 실제 최고점의 차이. Vanilla3DGS는 view 수 늘수록 gap 커짐(2-view +0.66dB → 12-view +1.94dB, 최대 ~2dB 보수적 하한). FSGS는 반대로 gap과 early-peak 비율이 함께 0에 가까워짐(12-view gap +0.14dB) — 300s가 사실상 FSGS의 정점이라 test leakage 우려가 적음.

## 5. 4개 기여 (서론 기준)

1. **품질--시간 Regime Map**: view×overlap×예산 격자에서 실용적 승패 영역 + Pareto frontier.
2. **동일 초기값 Refinement On/Off 통제 실험**: 모델 고정, optimization 순효과만 분리.
3. **기하 불확실성--렌더링 실패 연결 분석**: baseline/overlap 기하, densification 관측 부족, depth noise 개입.
4. **실용 가이드라인**: 조건별 선택 규칙.

(초기화 비교·OOD 역전 지점 대조는 리뷰 이후 이번 세션에 추가된 결과로, 서론 기여 목록에는 아직 별도 항목화되어 있지 않음 — §IV.5b/§IV.5c에 위치.)

## 6. 현재 완성도 (2026-08-20 기준, 세부는 `TASK_ASSIGNMENT_2026-08-20.md` 참고)

| 절 | 완성율 | 남은 것 |
|---|---:|---|
| 앞표지 | 85% | 이메일·전화·ORCID·교신저자 확정(최종선 vs 황인재 모순) |
| I–II 서론/관련연구 | 100% | 없음 |
| III 프로토콜 | 95% | Figure 1(프로토콜 개요 다이어그램) |
| IV 결과 | 95% | Gaussian 수/VRAM Pareto 정식 편입 |
| V 실패 분석 | 90% | 관측-부족 confound 분리(관측 부족 vs 배치 실패) |
| VI 가이드라인 | 90% | **overlap 축 본 실험 미실행**, 효율 축 편입 대기 |
| VII 결론 | 90% | dense-MVS 보조검증(낮은 우선순위) |
| 부록 | 60% | 부록 A(조기 종료, 러너 미작성) |

**Overlap 축은 co-visibility selector 검증만 끝났고, view×예산 규모의 본 실험은 아직 실행 전이다** — 서론의 3축 연구 질문 중 1/3이 아직 미실행 상태라는 뜻. 실행 스크립트는 준비 완료(`run_re10k_overlap_supplement.py`, `run_dl3dv_overlap_supplement.py`), 환경/입력 artifact는 2026-08-20에 정리해 커밋 완료.

## 7. 주요 한계 (VII절)

학습 view 분포 이탈(완전 제거 불가), confidence 출력의 optimization-계열 전용 비대칭, FSGS 초기화가 원 논문과 다름(sparse COLMAP), overlap 축 부분 적용, DL3DV pilot 규모(25-scene), 단일 GPU 순차 실행(~250 GPU-hour), RE10K 일부 community mirror 출처, 비교 범위가 단일-pass FF와 per-scene opt 두 축으로 제한(recurrent refinement인 ReSplat은 탐색적 참고로만 취급).

## 8. 외부 리뷰 요약 (2026-08-20 접수)

완성 전 상태 기준 KCI 8.5/10, 완성 시 9/10 근접 평가. 장점: 연구 질문 자체(방법 순위가 아니라 조건부 우위의 구조), Regime Map→Why→Intervention 구조, 방어적 통계 설계(peak-picking 금지, scene-cluster bootstrap), 가설 기각도 그대로 보고하는 태도, FF-init warm-start 발견을 논문의 "후반부 반전"으로 높이 평가. 우려: overlap 축 미실행이 가장 큰 리스크(연구 질문의 1/3), 결과가 과적재되어 있어(ReSplat/oracle-peak/early-stop 등) 척추(Regime Map→FF-init→메커니즘→가이드라인)를 흐릴 위험 — 부록/보조 증거로 격하 권장.
