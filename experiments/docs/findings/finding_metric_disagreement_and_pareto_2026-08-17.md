# 지표 불일치(PCC) + Gaussian/VRAM Pareto — 재실행 0 분석 (2026-08-17)

**실험 목적**: 새 실험 축을 늘리지 않고 이미 완료된 C1-a 결과(RE10K 2400행, DL3DV 2000행)에서
더 많은 지표를 뽑기 — 우선순위 2번(PCC, 지표 불일치)과 3번(Gaussian 수/VRAM Pareto).

**데이터/특징**: `metric_disagreement_pareto.py`가 `re10k_c1a_main`/`dl3dv_c1a_main`의 원본
per-run 로그(`logs/*.json` for FF, `{vanilla,fsgs}_runs/*/logs/*.json` for per-scene, seed
평균 후)를 직접 읽어 test_psnr/test_ssim/test_lpips/gaussian_count/peak_vram을 통합 테이블로
만든다. FF 방법(MVSplat/DepthSplat)은 test_ssim을 저장하지 않아 SSIM 비교는 FSGS-vs-Vanilla3DGS
쌍에서만 가능. 재실행 없음 — 기존 로그 파싱만.

---

## 1. 지표 불일치(PCC) 분석

**방법**: 3개 쌍(Vanilla3DGS-vs-FF, FSGS-vs-FF, FSGS-vs-Vanilla3DGS) × view_count마다, (scene,
budget) 셀 전체에 대해 ΔPSNR과 Δ(-LPIPS)(LPIPS는 낮을수록 좋아서 부호 반전) 사이 Pearson
상관계수(PCC), 그리고 두 지표가 가리키는 승자가 실제로 다른 비율(승자 불일치율)을 계산.

### 1.1 FSGS vs FF — 상관은 높은데 승자는 자주 뒤집힘 (반직관적 발견)

| Dataset | view | PSNR-LPIPS r | 승자 불일치율 |
|---|---|---|---|
| RE10K | 8 | +0.883 | 10.8% |
| RE10K | 12 | +0.862 | 12.5% |
| DL3DV | 8 | +0.935 | **33.0%** |
| DL3DV | 12 | +0.935 | **48.0%** |

DL3DV 12-view는 상관계수가 0.935로 매우 높은데도 승자가 거의 절반(48%) 뒤집힌다. 모순처럼
보이지만 실제로는 정합적이다 — 전체 분포에서는 두 지표가 같은 방향으로 움직이지만(그래서 r이
높음), FSGS와 DepthSplat이 이 regime(고view)에서 실력이 매우 비슷해서(§3 win/tie/loss 표의
8/12-view 근접 승부와 일치) 개별 셀에서는 근소한 margin이 지표에 따라 부호가 자주 바뀐다는
뜻이다. **즉 "지표가 서로 다른 결론을 낸다"가 아니라 "두 방법의 실력이 그 regime에서 지표
선택에 민감할 만큼 근접하다"는 신호로 읽는 게 맞다.**

### 1.2 FSGS vs Vanilla3DGS — LPIPS만 유독 다르게 움직임

| Dataset | view | PSNR-LPIPS r | PSNR-SSIM r | 승자 불일치율(PSNR/LPIPS) |
|---|---|---|---|---|
| RE10K | 2 | +0.276 | +0.853 | 64.2% |
| RE10K | 4 | +0.155 | +0.829 | 50.8% |
| DL3DV | 2 | +0.305 | +0.682 | 59.0% |
| DL3DV | 4 | +0.271 | +0.915 | 50.0% |

이 쌍은 저-view(2/4)에서 뚜렷한 패턴이 있다: **PSNR과 SSIM은 서로 잘 맞는데(r=0.68~0.92)
LPIPS는 거의 안 맞는다(r=0.16~0.31), 승자 불일치율도 50~64%로 매우 높다.** FSGS와
Vanilla3DGS 둘 다 저-view에서 품질이 전반적으로 낮은 구간이라(§3 PSNR 표 참고, 이 regime은
두 방법 다 10dB대) 지각 품질 지표(LPIPS)가 픽셀 단위 지표(PSNR)와 다르게 반응하는 것으로
보인다 — **이 regime(2/4-view, FSGS vs Vanilla3DGS)의 결과를 보고할 때는 PSNR만으로 승패를
말하지 않고 LPIPS도 같이 명시하는 게 안전하다.**

---

## 2. Gaussian 수 / VRAM Pareto (budget=300s)

| Dataset | view | Method | PSNR | Gaussians | Peak VRAM |
|---|---|---|---|---|---|
| RE10K | 12 | FSGS | 29.24dB | 109,818 | 2,109MB |
| RE10K | 12 | Vanilla3DGS | 21.38dB | 799,531 | 1,582MB |
| RE10K | 12 | MVSplat | 19.06dB | 786,432 | 7,299MB |
| DL3DV | 12 | FSGS | 30.07dB | 107,177 | 2,045MB |
| DL3DV | 12 | Vanilla3DGS | 22.90dB | 871,291 | 1,728MB |
| DL3DV | 12 | DepthSplat | 25.58dB | 1,376,256 | 6,773MB |

**FSGS는 고-view(8/12) regime에서 가장 높은 PSNR을 내면서 Gaussian 수는 다른 방법의
1/7~1/13 수준이다** — 명확한 온디바이스 효율성 우위. 다만 **peak_vram은 Gaussian 수 대비
불균형하게 높다**(RE10K 12-view: FSGS 2,109MB vs Vanilla3DGS 1,582MB인데 Gaussian 수는 FSGS가
7배 적음) — FSGS 학습 루프의 추가 손실 계산(depth correlation 등)이 VRAM을 더 쓰는 것으로
보인다. **"Gaussian 수는 적지만 VRAM은 그만큼 안 줄어든다"는 이 비대칭성 자체가 §5 온디바이스
논의에 넣을 만한 뉘앙스다.**

FF 방법(MVSplat/DepthSplat)은 view_count가 늘수록 Gaussian 수와 VRAM이 거의 선형으로
증가(view마다 고정 개수의 Gaussian을 예측하는 구조라 당연한 결과) — 8/12-view에서 VRAM이
6-7GB까지 치솟아 온디바이스 배포 관점에서는 FSGS/Vanilla3DGS보다 불리하다는 게 수치로
확인된다.

---

## 3. 제한 사항

- SSIM은 FF 방법 로그에 없어서 FSGS-vs-Vanilla3DGS 쌍에서만 비교 가능.
- peak_vram은 러너가 실행 시작 시 `torch.cuda.reset_peak_memory_stats()`를 한 번만 호출하고
  체크포인트마다 리셋하지 않으므로, budget=300s 시점 값은 "그 run 전체의 누적 최대"이지 그
  시점만의 순간 VRAM이 아니다(단일 Pareto 포인트로는 문제없지만, trajectory 내 여러 시점을
  비교할 때는 이 점을 감안해야 함).
- 재실행 없이 기존 데이터만 썼으므로 셀당 표본 수는 원래 C1-a 설계(scene 수)를 그대로 따른다.

## 4. 재현

```
python3 experiments/scripts/analysis/metric_disagreement_pareto.py
```

원자료: `experiments/outputs/metric_disagreement_pareto/rows.json`
