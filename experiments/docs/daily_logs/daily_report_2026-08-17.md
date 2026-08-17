# 일일 보고서 — 2026-08-17

## 오늘 목표

전날(2026-08-16) 저녁부터 이어진 세션. DL3DV C1-a 완료 확인 후, 우선순위 목록(overlap 축, PCC, Gaussian/VRAM Pareto, ReSplat 탐색적 비교)을 순서대로 처리하고, 논문의 남은 빈칸(정성적 비교 그림)을 채운다.

---

## 1. DL3DV C1-a 완료

**실험 목적**: 어제부터 배경에서 돌던 DL3DV C1-a 본 실험(25scene×4view_count×4budget×{DepthSplat, Vanilla3DGS×2seed, FSGS×2seed}) 완료 확인.

**결과**: 100/100 콤보, 2000/2000행, 총 35.3시간 소요. 최종 대시보드 갱신 후 재발행([링크](https://claude.ai/code/artifact/24c35b05-4da3-4877-a37b-bf38811fdbf5)).

**논문 연결**: RE10K와 함께 §Results의 메인 regime map/Pareto/win-loss 표의 데이터 기반 완성.

---

## 2. ReSplat 환경 구축 및 DL3DV 탐색적 확장 비교

**실험 목적**: recurrent refinement 계열(ReSplat, DepthSplat과 동일 계보)이 기존 4개 방법(MVSplat/DepthSplat, Vanilla3DGS, FSGS) 대비 어디에 위치하는지 확인 — 사전 등록 설계 밖의 탐색적 비교.

**데이터/특징**: conda env(`resplat`, Python 3.12) 신규 구축(torch 2.7.0+cu128, gsplat 1.5.3, pointops 빌드). DL3DV 체크포인트 3종(view8/16 base 256x448, RE10K view2 256x256) 다운로드. `resplat_dl3dv_runner.py` 신규 작성 — DepthSplat 러너와 같은 방식으로 encoder/forward_update/decoder를 직접 호출(자체 dataset loader 안 씀).

**버그 발견 및 수정**: 첫 스모크 테스트 PSNR 14~17dB(논문 기준 ~29dB) — 근본 원인은 `--overlap-summary` 기본값이 알려진 버그 있는 v1 view-selection 파일을 가리키던 것(DepthSplat 러너 코드를 본떠 만들다 그 스테일 기본값까지 같이 복사됨). v2로 수정 후 26.8dB로 정상화. 공식 데모(`infer_colmap.py` + COLMAP 데이터)로 환경/체크포인트 자체는 정상임을 먼저 확인한 뒤, 우리 데이터 로딩 경로만 별도로 디버깅해 원인을 좁혔다.

**결과**: DL3DV 25scene×4view_count 전체 실행(19.5분, 125행). 8-view(ReSplat 진짜 in-domain)에서 ReSplat 26.41dB로 4개 방법 중 최고(DepthSplat 대비 +0.6dB) — recurrent refinement의 실질 이득 확인. 12-view는 16-view 체크포인트로도 27.88dB로 FSGS(30.07dB)에는 못 미침 — 본 연구의 핵심 서사(고-view에서 per-scene optimization이 FF를 역전)가 더 강한 FF 기준선 아래에서도 유지됨.

**논문 연결**: `main.tex`에 새 절 "확장 비교: Recurrent Refinement (ReSplat)" 추가, Limitations 문구 업데이트. `finding_resplat_exploratory_comparison_2026-08-17.md`. 커밋 `002a3f9`, `49a1ba2`.

---

## 3. 지표 불일치(PCC) + Gaussian/VRAM Pareto — 재실행 0 분석

**실험 목적**: 새 실험 축 대신 이미 완료된 C1-a 로그에서 더 많은 지표를 뽑는다(어제 정한 우선순위 2, 3번).

**데이터/특징**: `metric_disagreement_pareto.py` 신규 작성 — RE10K/DL3DV의 원본 per-run 로그(FF `logs/*.json`, per-scene `{vanilla,fsgs}_runs/*/logs/*.json`)를 직접 파싱해 test_psnr/ssim/lpips/gaussian_count/peak_vram 통합.

**결과**: (1) FSGS-vs-FF 고-view에서 PSNR-LPIPS 상관은 매우 높은데(r=0.86~0.94) 승자 불일치율도 높음(DL3DV 12-view 48%) — 두 방법 실력이 근접하다는 신호로 해석. (2) FSGS-vs-Vanilla3DGS 저-view에서 LPIPS만 PSNR/SSIM과 다르게 움직임(r=0.16~0.31 vs 0.68~0.92). (3) FSGS는 고-view에서 최고 PSNR을 내면서 Gaussian 수는 다른 방법의 1/7~1/13 — 온디바이스 효율성 우위, 다만 VRAM은 비례해서 안 줄어듦.

**논문 연결**: `finding_metric_disagreement_and_pareto_2026-08-17.md`. 커밋 `002a3f9`(코드는 어제 밤, 이번 세션에서 실행·문서화는 계속).

---

## 4. 정성적 비교 그림 — RE10K + DL3DV(ReSplat 포함)

**실험 목적**: `main.tex`의 남은 `\todo{}` 중 재실행 비용이 거의 없는 것(이미 저장된 체크포인트 재렌더링)부터 채운다.

**데이터/특징**: `render_qualitative_gsplat.py`(GT/MVSplat/Vanilla3DGS, ps3 env) + `render_qualitative_fsgs.py`(FSGS, fsgs env) 신규 작성 — 저장된 체크포인트(`checkpoints/`, `vanilla_runs/`, `fsgs_runs/`)에서 동일 target camera로 재렌더링. RE10K scene `0588138dfec165a1`에서 8-view(역전 경계)·12-view(명확한 승패) 두 조건.

**스타일 개선**: 최신 NVS 논문 스타일로 다듬음 — 패널별 PSNR 라벨(최고값 강조), 자동 확대 인셋. 첫 시도는 GT-vs-MVSplat 오차 최대 영역을 크롭했더니 어둡고 텍스처 없는 구석이 걸려 다른 방법 인셋이 새까맣게 나옴 — GT gradient(디테일) × 오차로 스코어링하도록 수정해 창틀처럼 실제로 비교가 되는 영역을 찾게 고침.

**DL3DV+ReSplat 확장**: 같은 패턴을 DL3DV로 확장, ReSplat을 5번째 열로 추가(`render_qualitative_dl3dv_{gsplat,fsgs,resplat}.py`) — 5개 방법 모두 동일 target camera로 렌더링해 공정 비교. 결과가 집계 표와 일치(8-view ReSplat 27.99dB 1등, 12-view FSGS 30.26dB 1등).

**논문 연결**: `main.tex` §정성적 비교(GT/MVSplat/Vanilla3DGS/FSGS) + §확장 비교(5-way, ReSplat 포함) 둘 다 그림 채움. 커밋 `dca3df7`, `9aaa0ad`, `2386597`.

---

## 오늘 한 일 요약

- DL3DV C1-a 완료 확인, 최종 대시보드 갱신
- ReSplat 환경 구축 + PSNR 붕괴 버그(v1 view-selection 기본값) 발견·수정 + DL3DV 25scene 전체 실행
- PCC(지표 불일치) + Gaussian/VRAM Pareto 분석 — 재실행 0, 반직관적 발견 2건 문서화
- 정성적 비교 그림 2개(RE10K 4-way, DL3DV 5-way w/ ReSplat) 신규 생성, 논문 빈칸 채움
- 전부 커밋 완료(`09747af`~`2386597`, 총 8개 커밋)

## 다음 할 일

1. **C2(DTU depth-noise sensitivity) 착수** — 어제 우선순위에 밀려 멈췄던 실험. selector(`generate_dtu_overlap_candidates.py`, window_multiplier 버그 수정 완료) + runner(`--overlap-level` DTU 지원) 코드는 이미 준비돼 있음. 남은 건: depth cache precompute(`depth` env) + orchestrator 작성 + 실제 실행(σ noise×5 + scale bias×5, DTU 8scene×4 representative_conditions).
2. overlap 축 완성(RE10K/DL3DV 실제 high/low 비교 실행) — 규모 확정 필요(8scene 축소안 등, 어제 논의 중단됨).
3. C1-b(동일 초기값 refinement on/off)는 황인재 담당 — 진행 상황 확인 필요.
4. 실용 가이드라인(C4), 결론 요약 — 이제 결과가 많이 쌓였으니 초안 작성 시작 가능.
