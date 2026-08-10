# Model checkpoint / dataset domain table — 2026-08-10

이 표는 main benchmark를 고르기 위한 근거 문서다. 핵심 질문은 "데이터를 받을 수 있는가"보다 먼저, **feed-forward checkpoint가 그 데이터셋에서 in-domain인가**다.

## Feed-forward methods

| Method | Official checkpoint/domain evidence | Implication |
|---|---|---|
| MVSplat | 공식 README는 RE10K/ACID pretrained evaluation을 제공하고, DTU는 `RealEstate10K -> DTU` cross-dataset evaluation으로 안내한다. | Main benchmark에 MVSplat을 공정하게 넣으려면 RE10K가 가장 자연스럽다. DTU main 비교는 OOD penalty가 섞인다. |
| DepthSplat | MODEL_ZOO는 GS 모델이 RE10K and/or DL3DV에서 학습됐고, training views가 2~10 범위라고 명시한다. RE10K-only 2-view, DL3DV 2-6-view, RE10K+DL3DV 2-6/4-10-view checkpoint가 있다. | DL3DV는 DepthSplat 관점에서 매우 강한 main benchmark 후보. RE10K도 가능하지만 4/8/12-view 축은 checkpoint별로 재검토 필요. |

## Optimization methods

| Method | Domain issue | Implication |
|---|---|---|
| Vanilla3DGS | 장면별 최적화라 학습 도메인 OOD penalty는 없음. 대신 pose, mask/background, init 품질에 민감. | Dense-view sanity check로 pipeline 정상성을 먼저 확인해야 한다. |
| SparseGS/FSGS | 장면별 최적화 baseline. 구현/코드 안정성 확인 전 placeholder. | Main dataset 결정 후 붙인다. |

## Dataset candidates

| Dataset | Feed-forward domain match | Acquisition state | Current role |
|---|---|---|---|
| RE10K | MVSplat in-domain. DepthSplat도 RE10K checkpoint 있음. | `/data/Re-feem/datasets/re10k` 비어 있음. PixelSplat/MVSplat processed chunk 접근성 확인 필요. | Main benchmark 1순위 후보 |
| DL3DV | DepthSplat in-domain/near-domain. MVSplat은 직접 in-domain checkpoint 없음. | `/data/Re-feem/datasets/dl3dv` 비어 있음. HF access request 필요. | Main benchmark 강한 후보 |
| DTU | MVSplat/DepthSplat 모두 main in-domain 아님. | 공식 split 16개 확보 완료. | External validation / C2 / dense sanity |

## Official-source notes

- MVSplat README: RealEstate10K/ACID pretrained evaluation과 RE10K->DTU cross-dataset evaluation command를 제공한다.
- DepthSplat MODEL_ZOO: GS models are trained on RealEstate10K and/or DL3DV; training views range 2 to 10.
- DepthSplat DATASETS: view synthesis experiments mainly use RealEstate10K and DL3DV; RE10K follows pixelSplat/MVSplat 256x256 setup; DL3DV benchmark/test and 480P/960P paths are documented.
- DL3DV README: benchmark images+camera poses are ready on Hugging Face, and full 480P/960P images+poses releases are available but require access request and storage planning.

## Working decision

Do not declare DTU as main. Decide between:

1. **RE10K-first**: preserves MVSplat in-domain fairness. Best if MVSplat remains a core FF baseline.
2. **DL3DV-first**: better for DepthSplat and richer scene diversity. Best if DepthSplat becomes the primary FF representative.
3. **Two-small-subset probe**: download tiny RE10K and DL3DV subsets, run one FF model each, then decide. This is safest if download/auth effort is manageable.
