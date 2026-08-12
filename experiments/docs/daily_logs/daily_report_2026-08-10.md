# 일일 보고서 — 2026-08-10

## 오늘 목표

어제(8/9) 확보한 vanilla 3DGS·MVSplat 파이프라인이 "진짜 신호"를 내는지 검증하고, feed-forward 커버리지를 DepthSplat까지 넓히고, 실험 실행 구조를 아키텍처화하는 것. 동시에 다른 세션(task-5-14)의 병행 작업을 확인하고 git 상태를 정리했다.

---

## 1. 병렬 세션 작업 파악 + git 정리

- 같은 repo에서 작업 중인 다른 세션(`task-5-14`)이 이미 RE10K probe 다운로드, critical_path 재평가, DTU 공식 split 발견 등을 끝내놓은 것을 확인 — 중복 다운로드 정리.
- `git push`가 안 되던 두 가지 원인 발견: ① GitHub 인증 자체가 안 돼 있었음(계정 소유자가 직접 해결해야 함) ② 최근 커밋에 dense-sanity 체크포인트(325MB `.pt` 포함)가 실수로 딸려 들어가 GitHub 100MB 파일 제한에 걸림 — `.gitignore`에 `experiments/outputs_*/`, `*.pt`, `*.ckpt` 추가하고 히스토리 정리(`.git` 375MB → 708KB).
- 원격에 이미 `Lee`/`dev`/`CH`/`injai`/`main` 브랜치가 있는 것을 발견, 로컬에 `Lee`(작업 브랜치)·`dev`(통합 브랜치) 추적 브랜치 구성.

## 2. DTU dense-view sanity check 완주

scan1, 42-view(전체 49 중 held-out 7 제외), 30,000 iteration, budget 7200초 상한.

| 체크포인트 | iter | gaussians | PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|
| 1800s | 10,297 | 1,380,065 | 24.00 | - | 0.235 |
| 3600s (oracle) | 20,366 | 1,703,091 | **24.11** | 0.843 | 0.218 |
| 30k iter (최종) | 30,000 | 1,703,091 | 24.06 | 0.842 | 0.215 |

24.0~24.1dB로 수렴, 목표했던 "25dB 이상"에는 살짝 못 미치지만 SSIM·LPIPS는 명확히 건강. oracle 대비 최종이 미세하게 낮은 것도(24.11→24.06) sparse 8-view의 급격한 붕괴(10.69→9.85)와는 규모가 달라 — **파이프라인은 정상, sparse-view 저하는 진짜 현상일 가능성이 높다**는 결론.

## 3. MVSplat RE10K in-domain 검증

pixelSplat 공식 small subset에서 2-view 추론. **mean PSNR 25.6dB** (19.2~29.4dB). DTU zero-shot(4.8~11.8dB)과 대비되는 건강한 수치 — 카메라 변환 로직이 맞다는 근거.

## 4. DepthSplat 통합 + DL3DV in-domain 검증

- 공식 repo(`cvg/depthsplat`) clone, 격리 env 구축(torch 2.4.0+cu121), 커스텀 rasterizer 빌드.
- 체크포인트(`depthsplat-gs-base-dl3dv-256x448-randview2-6`) + 공식 2-scene test subset으로 검증.
- 아키텍처 채널 불일치(override 플래그 필요), encoder dict 반환(`return_depth=false` 필요) 두 가지 이슈 해결.
- context view를 DL3DV의 긴 walkthrough(410프레임)에서 너무 넓게 뽑으면 12dB, 좁게(30프레임 이내) 뽑으면 **mean PSNR 20.0dB**로 개선 — DL3DV 특유의 주의사항 발견.
- **feed-forward 2종(MVSplat/DepthSplat) + optimization 1종(Vanilla3DGS) 전부 실데이터 정상 동작 확인 완료.**

## 5. 아키텍처 일반화

- `model_registry.py`에 `conda_env_python`/`runner_script`/`external_repo`/`default_checkpoint` 필드 추가 — 모델별 실행 정보를 한 곳에서 관리.
- `run_experiment_batch.py` 신규 작성 — DTU 전용이던 `run_dtu_batch.py`를 일반화. `--dataset-root`로 데이터셋 무관, `--methods`는 registry lookup으로 자동 dispatch. 새 모델 추가 시 이 파일은 안 건드리고 registry만 갱신하면 됨.
- 3-way(Vanilla3DGS/MVSplat/DepthSplat) 배치로 검증 — DepthSplat은 정식 러너가 아직 없어 `no_runner`로 안전하게 skip되는 것까지 확인.

## 6. COLMAP 버그 수정 + 배치 마무리

어제 batch에서 scan30/103/110이 "LoadDatabase() 실패"로 죽던 문제 — 2-view 매칭이 0개일 때 COLMAP이 fatal check로 죽는 것이 원인. 매칭 0개 시 조용히 random-init fallback으로 넘어가게 수정, scan30/103/110 재실행으로 배치 완주 — **16개 공식 DTU scan × 2-view × seed0, Vanilla3DGS+MVSplat 전부 완주.**

## 7. 담당자 리뷰에서 나온 연구 설계 결정 (audit log §12)

- **densification on/off가 "공짜 통제 실험"이었다는 발견** — gsplat `DefaultStrategy`의 `refine_stop_iter=15,000` 때문에, dense-sanity 궤적의 iter 20,366~30,000 구간은 gaussian_count가 고정된 채(1,703,091) PSNR만 미세 하강(24.11→24.06)했다. 반면 어제 sparse 8-view 붕괴는 densification이 활발한 구간(iter<1749)에서 일어났다 — **두 하강이 같은 메커니즘이 아닐 수 있다.** → **결정: C1-b는 refinement on/off뿐 아니라 densification on/off까지 포함해서 돌린다** (H1 핵심 증거, ForeSplat 프로토콜과 비교 가능성 확보).
- **DTU 조명 혼입 가설 점검·기각** — dense-sanity 24dB/LPIPS 0.215가 "정상"보다는 "치명적 고장 없음" 수준이라는 지적에, 우리 다운로드 스크립트가 모든 scan·position에서 예외 없이 조명 index 3(`_3_`, 가장 diffuse)만 받았음을 확인 — 조명 혼입은 원인이 아님. 남은 후보(배경/마스크 미처리)는 DTU가 external-only라 우선순위 낮게 보류.
- **DepthSplat context 간격 결과(20↔12dB)의 재해석** — 같은 모델·같은 장면에서 view 선택만으로 8dB가 갈린 것은 (a) overlap이 세 통제 축 중 지배 요인일 조기 신호(V2), (b) 지금 probe의 context 선정이 co-visibility가 아니라 임의 프레임 간격이라 **교란 변수가 될 수 있다는 경고** — 정식 러너에서는 `generate_overlap.py`의 co-visibility 계산과 연결 필요(TODO로 기록).
- **공부방향.md 갱신** — B(gsplat 코드) 완료 처리 + 다섯 번째 발견(refine_stop_iter) 반영, D에 iteration축 15k 표시·Gaussian count 오버레이 추가, "신규" 항목(densification on/off)과 "병행" 항목(RE10K/DL3DV 확보는 이론 공부와 별도 트랙) 추가.

## 8. RE10K 획득 경로 재확인 + DL3DV 파일럿 확장

- README에 있던 "쉬운 전체 RE10K" 경로(pixelSplat 호스팅 서버 `schadenfreude.csail.mit.edu:8000`)가 **죽어있는 것을 확인**(connection refused) — 전체 RE10K는 여전히 YouTube 기반 다운로드+변환이라는 어려운 경로만 남아있음.
- DL3DV는 이미 접근 가능한 상태이므로 **DL3DV를 파일럿 규모로 먼저 확장**하기로 결정. `DL3DV-ALL-480P`의 11개 bucket(1K~11K)에서 spread 선정(seed=0)으로 20개 추가, 기존 probe 5개+신규 20개 = **25 scene, 1.9GB**.
- 25개 전부 구조 검증: `transforms.json`+`images_8` 정상, 네이티브 해상도 전부 동일(3840×2160), 다운샘플 배율 불일치 0건.
- **미확인 채로 남은 것**: 이 25개가 DepthSplat 공식 평가 split(`DL3DV-Benchmark`, 140 scene, 아직 미승인)과 겹치는지 — DTU 때와 같은 패턴("표준 split이 있는데 모르고 임의 선정")이 반복될 위험이 있어 SOURCE.md에 caveat로 남겨둠.

---

## 9. D 항목 실측(geometry uncertainty) + RE10K 재확장 + DL3DV 중복 확인

- **D 항목 1차 실측** (`geometry_uncertainty_figure.py`): scan1 dense-sanity COLMAP 재구성 861 pair에서 baseline·overlap·Gauss-Newton 기반 depth 불확실성을 실제 계산. baseline↔overlap(-0.87), baseline↔불확실성(-0.95)은 이론대로였지만 **overlap↔불확실성이 +0.95로 양의 상관** — baseline이 둘을 같은 방향으로 끄는 confounder로 보임. §5.3 서술에 영향 줄 수 있어 A-1 재독 후 재해석하기로 하고 기록(audit log §13).
- **RE10K 41→114 scene 확장**: pixelSplat/MVSplat과 같은 `.torch` chunk 포맷의 gate 없는 HF mirror(`Hualingchu/RealEstate10K_test`, 543 chunk)를 발견, 5 chunk만 받아 73 scene 신규 확보(중복 0). 새 데이터도 MVSplat으로 검증(22.4dB, 정상). **RE10K가 데이터 확보 병목에서 벗어남**(audit log §14.1).
- **DL3DV 25개 vs 공식 140-scene benchmark 중복 확인**: `DL3DV-Benchmark` 파일 목록(다운로드는 gate, 목록 열람은 가능)과 대조해 **중복 0** 확인 — 우리 25개는 train pool 쪽, 공식 eval set과 안전하게 분리(audit log §14.2).

## 10. 수식 정리 + 스크립트 디렉토리 재구성

- `experiments/docs/paper_gauss_newton_notation.md` 신규: Gauss-Newton 유도(담당자가 직접 함) 중 논문에 바로 쓸 부분만 추림 — 표기 대응표, $\text{Cov}(\hat\beta)\approx\sigma^2(J^\top J)^{-1}$ 핵심 수식, multi-view Jacobian이 코드(`two_view_depth_uncertainty`)와 대응되는 부분.
- `experiments/scripts/`를 `core/`(공유 모듈)·`runners/`(정식 러너)·`probes/`(1회성 검증)·`batch/`(driver)·`analysis/`(figure·overlap)로 재구성. 경로 참조 전부 수정 후 단위테스트 9개 + 주요 CLI 재검증 완료. README 등 stale 경로도 갱신.

## 종합 결론

1. **오늘로 "파이프라인이 작동하는가"라는 질문은 사실상 닫혔다.** Dense-view sanity check와 RE10K/DL3DV in-domain 검증이 서로 다른 각도에서 같은 결론(정상)을 가리킨다.
2. **다음 질문은 "왜 sparse에서 저하되는가"로 완전히 넘어갔다.** 이건 이제 실험보다 이론(공부방향.md의 A, Gauss-Newton/JᵀJ 조건수)이 앞장서야 하는 단계 — 단, densification on/off라는 새 통제 조건이 그 답을 구조적으로 도와줄 것.
3. **아키텍처가 정리돼서 다음 모델 추가 비용이 낮아졌다.** DepthSplat 정식 러너, SparseGS 통합이 남았지만 registry+batch 구조 덕분에 반복 작업이 아니라 registry 항목 추가로 끝난다.
4. **공유 repo 운영 리스크(인증, 대용량 파일, 브랜치)를 오늘 정리했다** — 이제 인증만 걸리면 push 가능한 상태.
5. ~~데이터 확보 축이 뒤집혔다~~ → 같은 날 후반에 해소(§9, audit log §14). RE10K mirror(`Hualingchu/RealEstate10K_test`) 발견으로 41→114 scene 확장, DL3DV(25)보다 오히려 많아짐. **RE10K가 더 이상 데이터 확보 병목이 아니다.**
6. **D 항목 1차 실측이 예상과 반대로 나왔다** — overlap과 depth uncertainty가 양의 상관(+0.95). baseline이 둘을 같은 방향으로 끄는 confounder라 그런 것으로 보이나, A-1 재독 완료 후 재해석 필요(audit log §13). §5.3 서술에 영향을 줄 수 있는 발견이라 오늘 결론에도 남겨둔다.

---

## 다음 실행 목록 (8/11로 이월)

1. □ **A-1 (Gauss-Newton/`JᵀJ` 재독)** — 진행 중. D의 이상 상관관계(overlap↑ uncertainty↑, §13) 해석이 이 재독에 달려있음.
2. ~~D 실측 figure~~ → 스크립트 작성+1차 실행 완료(`geometry_uncertainty_figure.py`, audit log §13). 예상과 반대 부호가 나와 **A-1 완료 후 재해석이 남은 작업**. iteration축 15k 표시/Gaussian count 오버레이는 아직 미착수.
3. □ **Densification on/off 통제 조건 구현** — `refine_stop_iter`를 낮추는 조건 하나 vanilla_3dgs_runner.py에 CLI로 추가, C1-b 설계에 반영.
4. □ **메인 데이터셋 최종 결정 (RE10K vs DL3DV)** — 이제 둘 다 데이터는 충분(114 vs 25). §12.3 기준(어느 feed-forward 모델이 in-domain인가)으로 순수하게 결정하면 됨.
5. ~~RE10K 전체 획득 경로 추가 탐색~~ → 완료. HF mirror로 114 scene 확보(§9).
6. □ **DepthSplat 정식 러너 작성** (`depthsplat_runner.py`, protocol_utils 스키마) — probe 스크립트에서 승격.
7. □ **View 선정을 co-visibility 기반으로 전환** — 지금 모든 probe/러너가 임의 프레임 간격으로 context를 고르고 있음. `generate_overlap.py`와 연결 필요.
8. ~~DL3DV 25개가 공식 DepthSplat-Benchmark split과 겹치는지 확인~~ → 완료, 중복 0(§9).
9. □ **E-1 (CS231A 노트 5, plane sweep)** — 황인재 트랙, MVSNet 읽기 전 필수.
10. □ **RE10K SOURCE.md의 출처(HF 개인 재업로드) 라이선스/인용 표기 정리** — 논문 집필 전에 확인.
