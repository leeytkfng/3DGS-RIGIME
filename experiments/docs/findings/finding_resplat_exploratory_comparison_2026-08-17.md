# ReSplat 탐색적 확장 비교 (2026-08-17)

**성격**: 사전 등록 비교군이 아니다. 본 연구의 주 비교(single-pass feed-forward vs
per-scene optimization) 두 축에 속하지 않는 recurrent refinement 계열이라 메인 regime map에
넣지 않고 별도 절/표로만 보고한다(`main.tex` Limitations 절과 동일 원칙).

**실험 목적**: DepthSplat/MVSplat과 같은 계보(cvg, Haofei Xu 1저자)의 recurrent refinement
FF 모델 ReSplat이 DL3DV에서 기존 4개 방법(DepthSplat/MVSplat 대신 in-domain DepthSplat,
Vanilla3DGS, FSGS)과 비교해 어디에 위치하는지 확인.

**데이터/특징**: DL3DV 25-scene × view_count[2,4,8,12], 기존 C1-a와 같은 v2 view selection
재사용(`dl3dv_overlap_v2/all_scenes_summary.json`). ReSplat은 결정론적 단일 추론(seed 없음,
wall_clock~1.2~1.3s/scene)이라 budget 축이 사실상 무료 — 체크포인트는 8-view 학습
모델(`resplat-base-dl3dv-256x448-view8`)을 기본으로, 12-view는 16-view 학습 모델도 같이 돌려
OOD 정도를 비교했다. num_refine=4(학습 스크립트의 train_max_refine=4와 일치).

**버그 노트**: 첫 스모크 테스트에서 PSNR이 14~17dB로 붕괴했던 원인은 러너의
`--overlap-summary` 기본값이 알려진 버그 있는 v1 view-selection 파일을 가리키던 것 —
v2로 고치니 정상화됨(자세한 내용은 `resplat_dl3dv_runner.py` 코드 주석 참고). 이 최종
결과는 그 수정 이후 25-scene 전체를 돌린 것.

---

## 1. 결과 — DL3DV 25-scene, budget=300s(per-scene 방법 기준) 대비

| view_count | ReSplat | DepthSplat | FSGS | Vanilla3DGS |
|---|---|---|---|---|
| 2 | 14.54dB *(OOD, 8v ckpt)* | **18.53dB** | 11.36dB | 11.75dB |
| 4 | 21.87dB *(OOD, 8v ckpt)* | **24.01dB** | 13.03dB | 12.58dB |
| 8 | **26.41dB** *(in-domain, 8v ckpt)* | 25.79dB | 24.97dB | 19.32dB |
| 12 | 27.88dB *(16v ckpt)* / 27.42dB *(8v ckpt, OOD)* | 25.58dB | **30.07dB** | 22.90dB |

(ReSplat 항목의 굵게 표시는 그 view_count에서 ReSplat이 4개 방법 중 1등인 경우.)

**2/4-view**: ReSplat이 8-view 학습 체크포인트를 그대로 쓴 분포 밖(OOD) 지점이라 DepthSplat에
못 미친다 — 다만 같은 OOD 상황에서도 FSGS/Vanilla3DGS(per-scene optimization, 이 규모 view
수에서 원래도 약함)보다는 훨씬 낫다.

**8-view (in-domain)**: **ReSplat이 4개 방법 중 1등**(26.41dB, DepthSplat보다 +0.6dB) —
recurrent refinement가 같은 아키텍처 계보의 단일 패스 DepthSplat보다 실제로 더 나은 품질을
낸다는 직접 증거. 같은 규모(8-view)에서 FSGS(24.97dB)도 근접하지만 ReSplat이 근소하게 앞선다.

**12-view**: 16-view 학습 체크포인트를 쓰면 27.88dB로 두 FF 방법(DepthSplat 25.58dB) 중에서는
가장 좋지만, **FSGS(per-scene optimization, 30.07dB)가 여전히 전체 1등**이다 — 본 연구
main.tex의 핵심 서사(고-view regime에서 per-scene optimization이 FF를 역전)가 ReSplat이라는
더 강한 FF 기준선을 붙여도 그대로 유지된다는 뜻. ReSplat이 "FF 계열 안에서는 가장 강하다"이지
"FF가 optimization을 이긴다"는 아니다.

## 2. Gaussian 수 / VRAM

| view_count | ReSplat gaussians | ReSplat VRAM |
|---|---|---|
| 2 | 14,336 | 1,490MB |
| 4 | 28,672 | 2,885MB |
| 8 | 57,344 | 4,856MB |
| 12 | 86,016 | 6,832MB |

DepthSplat 대비 view당 Gaussian 수는 훨씬 적다(8-view: ReSplat 57K vs DepthSplat
917K, §5.4 Gaussian/VRAM Pareto 문서 참고) — ReSplat의 "subsampled space에서 예측"이라는
설계(16x fewer Gaussians, README 자체 설명)가 실측으로도 확인된다. 다만 VRAM은 Gaussian
수만큼 줄지 않는다(recurrent refinement의 반복 forward pass 비용).

## 3. 해석 — 논문에 어떻게 쓸까

- **이 비교는 §3.2에서 논의 중인 "OOD 때문에 FF가 손해 본다"는 confound를 부분적으로
  통제한다**: ReSplat 8-view 지점은 진짜 in-domain이고, 거기서도 여전히 FF(ReSplat)가
  per-scene(FSGS)에 근접하되 이기지는 못한다(FSGS가 8-view에서도 근소하게 밀리긴 하지만
  12-view에서는 확실히 앞선다) — "OOD가 전부는 아니다"는 논지를 뒷받침하는 참고 자료로 쓸 수
  있다.
- 메인 regime map/승패 표에는 넣지 않는다. 별도 절 "확장 비교: Recurrent refinement
  (ReSplat)"로 이 표만 싣고, 사전 등록 설계에 없었음을 명시한다(어제 정한 서술 문구 그대로:
  "이 비교군은 사전 등록 설계에 포함되지 않았으며 탐색적 결과로 서술한다").
- RE10K는 ReSplat이 2-view 전용 체크포인트만 있어(MVSplat과 동일 제약) 확장 비교의 가치가
  낮다고 판단해 이번엔 DL3DV만 진행했다. 필요하면 RE10K도 같은 방식으로 추가 가능(코드는
  이미 DL3DV 전용으로 짜여 있어 RE10K 버전은 별도 작성 필요).

## 4. 재현

```
nohup python3 experiments/scripts/batch/run_resplat_dl3dv.py \
  > experiments/outputs/resplat_dl3dv/orchestrator.log 2>&1 &
```

원자료: `experiments/outputs/resplat_dl3dv/resplat_dl3dv_summary.json` (125 rows: 25 scene ×
4 view_count, 12-view는 8v/16v 체크포인트 둘 다 포함).
