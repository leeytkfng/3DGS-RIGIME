# 논문 2차 완료안 이후 작업 분배 (2026-08-20)

**배경**: C2(depth-noise sensitivity, DTU 8-scan) 640/640 combo 완주, 리뷰 반영(초기화 비교·OOD
역전 지점 대조 등 신규 분석 포함) 완료 후 시점의 작업 분배. 전체 완성율은 본문 기준 약 88%
— 남은 건 대부분 GPU 실행 대기 항목과 이미 있는 데이터를 정식 절로 편입하는 작업이다. 1

---

## 0. C4-a 처리 결정

**논문 밖으로 유지.** MVSplat이 학습 목적과 무관하게 이미 출력하는 opacity가 동일 초기값
refinement 중 Gaussian 이동량과 약하게 상관된다는 파일럿 결과(37 scene, 방향 일관 89%)를
얻었으나, densification이 켜진 상태라 iter0→60s 대응이 nearest-neighbor 근사이고 아직 검증
전이다. §VII 한계에 한 줄만 언급(비공개 내부 문서로 지칭)하고 본문에는 넣지 않는다.

- 설계·파일럿 결과: `experiments/docs/C4A_DESIGN_2026-08-19.md`
- 파일럿 스크립트: `experiments/scripts/analysis/c4a_offtheshelf_uncertainty_pilot.py`
- 다음 단계(후속 과제, 이번 논문 범위 아님): `--densification off` 소규모 재실행으로 근사 여부
  확인 → 확인되면 별도 논문/후속 절로 확장 검토.

---

## 1. 황인재(2저자) — GPU 실행 위주

| 작업 | 상태 | 비고 |
|---|---|---|
| **Overlap 축 본 실험** | 스크립트 완성, 실행만 하면 됨 | `experiments/scripts/batch/run_re10k_overlap_supplement.py`, `run_dl3dv_overlap_supplement.py`. GPU 현재 유휴(C2 완료). RE10K/DL3DV 각 8-scene(seed=0 결정론적 서브샘플) $\times$ 4 view_count $\times$ {high, low} overlap, seed=0 단일. 예상 ~11–12시간 |
| **부록 A 조기 종료 보조실험** | 프로토콜은 이미 정의됨, 러너 신규 작성 필요 | `experiments/docs/paper/overleaf_draft/sections/03_protocol.tex` §III.10(``조기 종료 규칙'')이 정의: 기존 context view 1개를 held-out validation으로 제외, budget snapshot 촘촘히 찍어 validation PSNR이 최종값의 95%에 처음 도달하는 시점 $t_{95}$ 계산. `vanilla_3dgs_runner.py`의 기존 budget-snapshot 로직 재사용 가능 |
| (여유 있으면) C4-a `--densification off` 검증 | 논문 밖, 후속 과제 | §0 참고. 2-view 몇 scene이면 충분 |

## 2. 서창희(3저자) — 분석/작성 위주, GPU 불필요

| 작업 | 상태 | 비고 |
|---|---|---|
| **Gaussian 수/VRAM Pareto §IV 정식 편입** | 데이터 이미 있음(zero-rerun) | `experiments/scripts/analysis/metric_disagreement_pareto.py` 결과(`experiments/outputs/metric_disagreement_pareto/rows.json`)를 `04_results.tex`에 정식 서브섹션으로 — 표/그림/본문 작성. `06_guideline.tex` §VI.2가 이미 이 편입을 전제로 쓰여 있어 완료 후 그 문단을 참조로 교체 |
| **관측-부족 confound 분리 측정** | 새 분석 스크립트 필요 | `05_failure_analysis.tex` §V.2의 `\chk` 참고: "관측 부족"(카메라 수 적어 소수 view에서만 관측)과 "배치 실패"(random-sphere fallback 점이 애초에 어떤 카메라 절두체에도 안 잡힘)를 분리해야 함 — 어떤 view에도 투영되지 않는 Gaussian 비율을 기존 체크포인트+카메라 파라미터로 별도 측정(GPU 불필요할 가능성 높음) |
| **Figure 1 프로토콜 개요 다이어그램** | 미착수 | `03_protocol.tex` 도입부 자리. 통제 축 3개(view/overlap/budget) + 서사 사슬 4단계, 3~4개 박스+화살표 수준(같은 파일의 `\chk` 참고) |

## 3. 이용수 + Claude — 계속 진행

- **저자 정보/교신저자 확정**: 최종선 교수님께 직접 확인 필요(‡ 표시가 저자줄엔 최종선, 각주엔
  황인재로 모순 — `00_front.tex`).
- **논문 분량 10–11페이지 조정**: 아직 미착수. Overleaf 컴파일해서 실제 페이지 수 확인 후 자간·
  줄간격·문단 압축으로 조정(과하게 줄이지 않는 선에서).
- 전체 통합·최종 검수.

---

## 4. 참고 — 완성율 스냅샷(2026-08-20 기준)

| 절 | 완성율 | 남은 것 |
|---|---:|---|
| 앞표지 | 85% | 이메일·전화·ORCID·교신저자(§3) |
| I–II. 서론/관련연구 | 100% | 없음 |
| III. 프로토콜 | 95% | Figure 1(§2) |
| IV. 결과 | 95% | Pareto 편입(§2) |
| V. 실패 분석 | 90% | 관측-부족 confound 분리(§2) |
| VI. 가이드라인 | 90% | overlap 축(§1), 효율 축 편입 대기 |
| VII. 결론 | 90% | dense-MVS 보조검증(낮은 우선순위) |
| 부록 | 60% | 부록 A(§1) |
