# 일일 보고서 — 2026-08-14

## 오늘 목표

어제 밤에 착수한 C1-a 본 실험(RE10K 30-scene)이 백그라운드에서 밤새 도는 걸 확인하는 것부터 시작. 이어서 병행 가능한 작업(DL3DV 밀린 것들, overlap_level 축 준비, SplatFormer 인용 확인)을 처리한다.

---

## 1. FSGS 300초 budget 붕괴 버그 발견 및 수정

**실험 목적**: 밤새 돈 C1-a 본 실험의 중간 결과를 확인하던 중 발견. 원래는 단순 진행 상황 점검이었는데, 실제 숫자를 뽑아보니 심각한 이상 패턴이 나와 즉시 원인 추적으로 전환.

**데이터/특징**: C1-a 본 실험 50/120 combo 완료 시점(총 1000 rows)의 pooled 평균 표를 뽑아봤다. 8-view/12-view 조건에서 FSGS의 budget 300초 결과가 60초 대비 급격히 나빠지는 패턴이 보였다(예: 12-view 27.5dB→14.4dB). 확인한 scene 6개 전부(8-view, 12-view, 2-view, 4-view 각 데이터 포함) 동일하게 재현됐다 — 특정 scene의 우연이 아니라 100% 체계적인 버그.

**쉽게**: FSGS 코드를 wall-clock 시간 기준으로 돌게 우리가 고쳐서 썼는데, 그 안에 숨어있던 원본 코드 조각 두 개가 문제였다. 하나는 "10,000번 반복 넘으면 더 이상 학습 안 함"(파라미터 업데이트가 멈춤 — gradient는 계산하는데 실제로 반영을 안 함), 다른 하나는 "3000번마다 opacity를 초기화"(이건 10,000번 넘어도 계속 발동). 원래 FSGS는 정확히 10,000번만 도니까 이 두 코드가 서로 부딪힐 일이 없었는데, 우리는 300초 동안 최대 5만 번 넘게 도니까 10,000번을 훌쩍 넘긴 뒤로 "opacity는 계속 초기화되는데 회복할 방법(학습)이 없는" 상태가 반복되면서 결국 거의 다 투명해진 채로 끝나버린 것.

**전문 용어**: (1) `if step < opt.iterations: gaussians.optimizer.step()` — `opt.iterations=10,000`(FSGS 원본 기본값)이 그대로 남아있어서, 300초 budget에서 실제 도달하는 iteration(4~5만대)을 훌쩍 넘김에도 그 이후로는 `optimizer.step()`이 전혀 호출 안 됨. (2) `opacity_reset_interval=3000`이 상한 없이 계속 발동, 게다가 `densify_until_iter=10,000` 이후로는 `densify_and_prune`도 이미 멈춘 상태라 reset을 상쇄할 재성장 메커니즘도 없음. 두 버그가 겹쳐 10,000 iteration 이후 opacity가 여러 번 초기화되고 한 번도 회복 못 함.

**수정**: `experiments/scripts/runners/fsgs_runner.py` — optimizer step은 항상 실행(우리 stopping 기준은 `opt.iterations`가 아니라 elapsed/`--max-iterations`), opacity reset은 `step < opt.densify_until_iter` 조건 추가(densification이 살아있는 동안만 — 원 3DGS/FSGS 설계 의도와 일치).

**검증**: 가장 심하게 무너졌던 조건(scene `6771a51bf0cfce7f`, 12-view, seed0)으로 재실행. 60s→300s가 버그 전엔 23.58dB→14.46dB(붕괴)였는데, 수정 후엔 22.00dB→22.46dB(정상적으로 소폭 개선)로 바뀜. train_loss도 0.0085로 정상적으로 낮게 수렴.

**후속 조치**: 오염된 FSGS trajectory 로그 100개(50 combo × seed 2) 전부 삭제, Vanilla3DGS(101개)·MVSplat(50개) 로그는 이 버그와 무관해 보존. 같은 명령으로 재실행 → 재개 로직이 정확히 FSGS만 다시 돌리는 것 확인(스킵 로직 실측 확인: 새 프로세스가 곧바로 `fsgs_runner.py`부터 시작, `vanilla_3dgs_runner.py`/`mvsplat_re10k_runner.py`는 재호출 안 됨). 손실 시간 재계산: 총 경과 17.7시간 중 절반(Vanilla3DGS/MVSplat 몫, ~8.6시간)은 보존, FSGS 몫(~8.6시간)만 재작업 — "처음부터 다시"가 아니라 "절반만 다시"로 확정.

**논문 연결**: 이건 C1-a의 가장 중요한 조건(budget=300s, "시간을 충분히 주면 optimization이 이기는가"라는 핵심 질문)을 직접 오염시키는 버그였다 — 발견을 늦게 했으면 본 실험 전체가 무효가 될 뻔했다. `overall.md`나 논문 초안의 FSGS 관련 서술에는 영향 없음(방법론 자체의 문제가 아니라 우리 wall-clock 어댑테이션 과정의 구현 버그였음을 명확히 구분).

---

## 2. DL3DV 통제 스모크 — 세 방법론 파이프라인 최종 검증

**실험 목적**: RE10K 본 실험이 끝나면 바로 DL3DV 본 실험에 착수하기로 했다(사용자 결정). 그 전에 DL3DV 파이프라인이 최종 설정(고친 view 선택 v2, FSGS 포함 3-way)으로 실제 돌아가는지 마지막 확인.

**데이터/특징**: DL3DV scene `09b05fa3...`, `dl3dv_overlap_v2`(DepthSplat 실제 알고리즘 재현한 고친 view 선택) 기준. RE10K 본 실험이 GPU를 쓰는 동안 짧은 budget(1s/10s)으로만 병행 실행 — 본 실험에 부담 안 주려고 최소 규모로.

**결과**: Vanilla3DGS(18.6dB), FSGS(18.7dB), DepthSplat(2-view in-domain, 20.1dB) 전부 정상 실행·정상 범위 수치. RE10K와 동일한 세 방법론 구조가 DL3DV에서도 그대로 동작함을 확인.

**논문 연결**: `run_dl3dv_c1a_main.py`(아직 미작성, `run_re10k_c1a_main.py` 패턴을 그대로 옮기면 됨) 착수 전 마지막 파이프라인 검증이 끝났다. RE10K 본 실험 완료 후 바로 이어서 DL3DV 본 실험 착수 예정.

---
