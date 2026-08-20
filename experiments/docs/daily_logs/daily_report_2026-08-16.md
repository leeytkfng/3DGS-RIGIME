# 일일 보고서 — 2026-08-16

## 오늘 목표

DL3DV C1-a 본 실험이 배경에서 계속 도는 동안, 황인재가 보낸 "B 인계 대응" 작업 지시(A-1/A-2)를 처리하고, 이후 사용자의 우선순위 재조정에 따라 새 실험 축(C2 DTU) 준비를 잠정 보류하고 기존 데이터 기반 분석/실험 완성으로 방향 전환.

---

## 1. A-1: 60dB gate 오해 해소

**실험 목적**: 황인재가 "60dB gate 미달로 C1-b 판정 보류"라는 취지의 작업 지시를 보내옴 — 우리 쪽 실제 동결 기준과 대조해서 진위를 확인해야 했다.

**데이터/특징**: `git log --oneline --all -S "renderer_equivalence_tolerance"` 전체 히스토리 재현 — 이 파일에서 해당 값을 건드린 커밋은 `97a5615`(legacy MSE 0.0001)와 `6884922`(2026-08-13, 390샘플 실측 근거로 PSNR≥33dB 최종 동결) 딱 둘뿐. 저장소 전체에서 "60dB" 문자열 검색해도 gate 기준으로 쓰인 곳은 전무.

**결과**: 60dB는 이 저장소 히스토리 어디에도 존재한 적 없음을 확정. B의 43.8~44.3dB 측정값은 우리 33dB 기준을 여유 있게 통과(PASS). 다만 B와 측정 방식(view 집합/코드/체크포인트)이 동일한지는 컨테이너 분리로 확인 불가 — 최종 통보는 보류.

**논문 연결**: `experiments/docs/GATE_RESOLUTION_2026-08-16.md` 작성, 커밋(`09747af`). 팀 커뮤니케이션 문서일 뿐 논문 본문에는 직접 안 들어감.

---

## 2. A-2: target-coverage confound 진단

**실험 목적**: 같은 작업 지시의 두 번째 항목 — `core/view_selector.py`의 high/low overlap 후보가 overlap 축과 무관하게 "target을 얼마나 잘 둘러싸는가"도 같이 바꾸는지(=selector 자체의 confound) 확인.

**데이터/특징**: RE10K 30scene + DL3DV 25scene 전체, view_count 2/4/8/12 × high/low에 대해 bracketing(convex hull 포함 여부), 최소 각도 차, target-context 거리를 계산. `scene_cluster_bootstrap_ci`로 high/low 및 paired delta의 신뢰구간 산출.

**결과**: bracketing은 두 데이터셋 다 단일 궤적 forward-facing 촬영이라 거의 안 일어나서(0~20%) 뚜렷한 신호 없음. 대신 **min_angle_deg/nearest_dist에서 강하고 일관된 confound 발견** — view_count≥4 거의 모든 조건에서 low overlap 후보가 high보다 target에 각도상·거리상 유의하게 더 가까움(예: DL3DV 8-view 각도차 +31.5° [21.9,42.0], CI가 0을 완전히 벗어남). 원인: high 후보는 pool 안 무작위 위치의 좁은 window에서만 뽑는데, low 후보는 pool 전체(target 인접 구간 포함)에서 뽑아서 구조적으로 target 근처를 우연히 더 잘 줍는다.

**중요한 점**: 이 selector는 이번 세션에 새로 만든 모듈이라 지금까지 실측된 C1-a 본 실험(RE10K 2400행+DL3DV 진행분)에는 전혀 쓰인 적 없음 — 기존 결과는 안전. 앞으로 이 selector로 overlap 축 실험을 돌릴 때만 해당하는 경고. 코드(`view_selector.py`)는 건드리지 않음 — 진단만 하고 변경은 공동 승인 필요라는 원 지시 그대로 따름.

**논문 연결**: `experiments/docs/TARGET_COVERAGE_CONFOUND_A_2026-08-16.md`, 스크립트 `experiments/scripts/analysis/target_coverage_confound.py`, 원자료 `experiments/outputs/target_coverage_confound/rows.json`. 전부 커밋(`09747af`).

---

## 3. DL3DV C1-a 대시보드 정기 갱신

**실험 목적**: 시간 단위 크론으로 진행 상황 대시보드를 최신 상태로 유지.

**데이터/특징**: `c1a_main_summary.json` 파싱해 진행 콤보/행 수 확인 → `generate_regime_map.py --ff-method DepthSplat --dataset-label DL3DV`로 그림 갱신 → 대시보드 HTML의 ROWS/SNAPSHOT_AT 패치 → 재발행.

**결과**: 오늘 안에 62/100 → 71/100 콤보(1420/2000행)까지 진행. 아직 미완료 — 계속 배경에서 돎.

**논문 연결**: [DL3DV 대시보드](https://claude.ai/code/artifact/24c35b05-4da3-4877-a37b-bf38811fdbf5) 실시간 트래킹용, 논문 본문과는 별개.

---

## 4. C2(DTU depth-noise sensitivity) 준비 — 착수 후 보류

**실험 목적**: DL3DV 끝나고 GPU 빌 때 바로 착수할 수 있도록 C2 실험(σ noise×5, scale bias×5, DTU 8scene×4 representative_conditions) 코드를 미리 준비해두라는 사용자 지시(오전)에 따라 시작.

**데이터/특징**: `generate_dtu_overlap_candidates.py` 신규 작성 — MVSplat 공식 DTU 16-scan 풀에서 seed=0 결정론적 샘플링으로 8scan 선정(scan 1/8/30/31/34/38/40/114), 2/4/12-view high/low context 후보 생성.

**결과 — 실행 중 실제 버그 발견**: `view_selector.py`의 기본 window_multiplier=4.0을 그대로 쓰니 DTU에서 12-view "high" 조건의 window 크기(48)가 실제 pool 크기(42)를 넘어서 **window가 사실상 pool 전체가 되어버려 high와 low가 거의 같아지는 문제** 발견(방향검증 16/24만 통과, 12-view에서 집중 실패, 수치까지 거의 동일하게 나옴). DTU 전용으로 window_multiplier를 2.0으로 낮춰 재실행하니 22/24로 정상화(12-view 전부 통과, 남은 2개 실패는 4-view의 단일 seed 노이즈로 보임).

병행해서 `vanilla_3dgs_runner.py`/`precompute_depth_maps.py`에 DTU용 `--overlap-level` 지원 추가(RE10K 패턴 그대로 이식).

**중요한 점**: 이후 사용자가 "새 축을 늘리지 말고 이미 돌린 결과에서 지표를 더 뽑으라"는 방향으로 우선순위를 재조정 — C2 DTU orchestrator 마무리/실제 실행은 여기서 멈춤. 지금까지 만든 코드(selector 적용 스크립트, runner 확장, window_multiplier 버그 수정)는 남겨두되, 다음 우선순위로 다시 올라올 때 이어서 진행.

**논문 연결**: 아직 없음(orchestrator 미완성, 실행 전). 코드는 커밋 안 됨(작업 중 상태) — 내일 정리해서 커밋 필요.

---

## 5. 우선순위 재조정 — "축 늘리지 말고 기존 결과에서 더 뽑기"

**배경**: 사용자가 ReSplat(recurrent refinement 계열, cvg/resplat) 확장 비교 가능성을 검토한 긴 메모를 공유 — 데이터 포맷 호환(같은 연구실/1저자, RE10K chunk 포맷 공유)과 GPU 비용이 거의 안 든다는(추론 0.8초, 결정론적, 30scene×4view×1회≈2분) 분석. 다만 이건 **C1-a 다 끝난 뒤 검토할 사안**으로 명시 — 지금 착수 대상 아님. 메인 regime map에는 안 넣고 별도 절/표로 "탐색적 확장 비교"로 서술하기로 방향만 정해둠(사전 등록 비교군 아님을 숨기지 않는다는 원칙).

**실제 오늘의 지시**: "축을 늘리지 말고, 이미 돌린 결과에서 지표를 더 뽑아라." 우선순위:
1. **overlap 축 완성** — 원래 계획에 있었는데 안 한 것(새 축 아님). RE10K는 MVSplat 러너에 이미 `--overlap-level` 있었고, DL3DV 쪽(Vanilla3DGS/FSGS/DepthSplat)에는 오늘 새로 배선함.
2. **PCC 추가** — 조사해보니 논문에서 말하는 "지표 불일치 분석"(`overall.md` §366/§403에 이미 계획돼 있던 항목)과 같은 것으로 판단됨. 이미 저장된 test_psnr/ssim/lpips 스칼라 간 상관계수를 내는 거라 진짜 재실행 0으로 가능(픽셀 단위 PCC였다면 재렌더링이 필요해서 얘기가 달랐을 것).
3. **Gaussian 수/VRAM Pareto** — 조사 결과 5개 러너(Vanilla3DGS/FSGS/MVSplat×2/DepthSplat) 전부 `gaussian_count`/`peak_vram`을 매 로그마다 이미 저장하고 있음을 실제 로그 파일로 확인. 정말 재실행 0.
4. **FPS** — 저장된 데이터 어디에도 없음(wall_clock은 학습 시간이지 렌더 속도 아님). 하려면 저장된 체크포인트에서 가벼운 timed forward pass가 새로 필요 — 완전한 재실행 0은 아님. 우선순위 가장 낮음.

**진행 상황**: overlap 축 완성의 실행 규모(전체 30/25scene×4view×2level 기준 RE10K 44시간/DL3DV 37시간)를 계산해서 보고했고, 축소안(8scene 등)을 제시하려던 중 세션이 끊김 — **내일 이어서 규모를 확정해야 함.**

**논문 연결**: PCC/Gaussian-VRAM Pareto는 §5.4 데이터 기반이라 스크립트만 작성하면 바로 결과 나옴 — 내일 우선 착수 후보.

---

## 오늘 한 일 요약

- A-1(60dB gate 오해 해소), A-2(target coverage confound 진단) 완료 및 커밋(`09747af`)
- DL3DV 대시보드 정기 갱신 계속(62→71/100 콤보)
- C2(DTU) 준비 중 window_multiplier 구조적 버그 발견·수정, DTU overlap-level 러너 지원 추가 — 이후 보류(미커밋)
- DL3DV 러너 3종(Vanilla3DGS/FSGS/DepthSplat)에 overlap-level 지원 배선 완료(미커밋)
- PCC/Gaussian-VRAM Pareto/FPS 재실행 필요 여부 조사 완료 — PCC와 Gaussian/VRAM은 재실행 0으로 확인, FPS는 새 경량 계산 필요

## 내일 할 일

1. overlap 축 보조 실험 규모 확정(8scene 축소안 등) 후 착수
2. PCC(스칼라 지표 상관계수) 분석 스크립트 작성·실행 — 재실행 0, 바로 가능
3. Gaussian 수/VRAM Pareto 분석 스크립트 작성·실행 — 재실행 0, 바로 가능
4. 오늘 미커밋 상태로 남은 러너 변경사항(vanilla_3dgs_runner.py, fsgs_runner.py, depthsplat_dl3dv_runner.py, precompute_depth_maps.py, generate_dtu_overlap_candidates.py) 정리 후 커밋
5. DL3DV C1-a 완료 확인(100/100) 시 최종 대시보드 갱신 + 크론 종료
6. ReSplat 확장 비교는 C1-a 완료 후 검토(오늘은 착수 안 함, 방향만 기록)
