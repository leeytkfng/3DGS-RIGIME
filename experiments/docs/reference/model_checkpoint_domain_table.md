# Model Checkpoint / Dataset Domain Table — updated 2026-08-11

이 표는 main benchmark를 고르기 위한 근거 문서다. 핵심 질문은 "데이터를 받을 수 있는가"가 아니라, **feed-forward checkpoint가 해당 데이터셋에서 in-domain인가**다. 2026-08-10 기준 RE10K와 DL3DV 모두 pilot 규모 이상 확보됐으므로, 이제 선택 기준은 공정성이다.

## Feed-forward methods

| Method | Official checkpoint/domain evidence | Local validation | Implication |
|---|---|---|---|
| MVSplat | 공식 flow는 RE10K/ACID pretrained evaluation을 중심으로 하고, DTU는 `RealEstate10K -> DTU` cross-dataset evaluation으로 안내된다. | RE10K 2-view probe mean PSNR 25.6dB, 추가 mirror chunk 22.4dB. DTU zero-shot은 4.8~11.8dB로 OOD penalty가 컸다. | Main benchmark에 MVSplat을 공정하게 넣으려면 RE10K가 가장 자연스럽다. DTU main 비교는 피한다. |
| DepthSplat | MODEL_ZOO는 RE10K and/or DL3DV 학습 checkpoint를 제공한다. DL3DV GS base checkpoint는 256x448, random view 2~6 계열이다. | DL3DV near-context probe mean PSNR 20.0dB. 넓은 context에서는 약 12dB라 view selection 민감도가 크다. | DL3DV는 DepthSplat in-domain 후보. RE10K도 checkpoint 확인 후 main에 넣을 수 있다. |

## Optimization methods

| Method | Domain issue | Local validation | Implication |
|---|---|---|---|
| Vanilla3DGS | 장면별 최적화라 학습 도메인 OOD penalty는 없다. 대신 pose, mask/background, init, 입력 해상도 protocol에 민감하다. | DTU dense-view scan1, 42 train view, 30k iter에서 PSNR 24.0~24.1dB, SSIM 0.843, LPIPS 0.218. | 배관은 정상. Main에서는 feed-forward와 같은 해상도(예: RE10K 256x256)를 반드시 사용해야 한다. |
| SparseGS/FSGS | 장면별 최적화 baseline. 구현/코드 안정성 확인 전 placeholder. | 아직 미통합. | Main dataset 확정 후 통합한다. |

## Dataset candidates

| Dataset | Feed-forward domain match | Acquisition state | Current role |
|---|---|---|---|
| RE10K | MVSplat in-domain. DepthSplat도 RE10K 계열 checkpoint가 있어 확인 대상. | `/data/Re-feem/datasets/re10k`: test 8 chunk, 114 scene, 약 1.2GB. `SOURCE.md` 작성됨. | **Main benchmark 1순위 후보.** 256x256 고정. |
| DL3DV | DepthSplat in-domain. MVSplat은 별도 in-domain checkpoint가 없으면 OOD. | `/data/Re-feem/datasets/dl3dv`: 25 scene, 1.9GB, official benchmark 140 scene과 중복 0. | DepthSplat 중심 보조 main 또는 second benchmark 후보. 256x448 track 가능. |
| DTU | MVSplat/DepthSplat 모두 main in-domain 아님. | `/data/Re-feem/datasets/dtu`: 공식 split 16개 포함 총 29 scan. | External validation / C2 / dense sanity. |

## Resolution rule

- RE10K/pixelSplat/MVSplat line: 256x256 protocol을 상속한다.
- DepthSplat DL3DV checkpoint: 256x448 protocol을 상속한다.
- Per-scene optimization baseline도 같은 이미지 해상도와 같은 train/test view split을 사용한다. 해상도를 다르게 주면 FF vs optimization 비교가 무효가 된다.

## Working decision

현재 상태에서는 **RE10K-first**가 가장 보수적이다. 이유는 MVSplat의 in-domain 조건이 명확하고, 확보된 114 scene이 본 실험 규모를 이미 넘으며, DTU에서 확인한 OOD penalty 교란을 피할 수 있기 때문이다. DL3DV는 DepthSplat 중심 보조 benchmark로 유지하되, MVSplat을 포함할 경우 OOD임을 분리 표기해야 한다.

## 남은 확인

1. DepthSplat RE10K checkpoint 존재/지원 view 수를 실제 repo checkpoint 목록과 로컬 파일 기준으로 확정한다.
2. MVSplat 4/8/12-view 실행 가능 여부와 confidence 출력 유무를 확인한다.
3. DepthSplat DL3DV 2/4/6-view 또는 2/4/8/12-view mapping을 runner 수준에서 확정한다.
4. RE10K mirror의 license/citation 표기를 논문용 문구로 정리한다.
