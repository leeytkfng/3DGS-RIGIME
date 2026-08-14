# Dataset recommendation for the sparse-view 3DGS regime study

## 2026-08-10 update

이 문서는 초기 권장안을 대체한다. 현재 가장 중요한 기준은 단순 다운로드 편의가 아니라 **feed-forward 공개 checkpoint의 학습 도메인과 main benchmark를 맞추는 것**이다.

## 핵심 원칙

DTU는 main benchmark로 쓰지 않는다. DTU에서 MVSplat은 RE10K-trained checkpoint의 cross-dataset/OOD evaluation이 되며, per-scene optimization은 장면별로 새로 맞추므로 동일한 OOD penalty를 받지 않는다. 따라서 DTU main comparison은 패러다임 비교가 아니라 domain-shift 비교로 읽힐 위험이 크다.

## Candidate datasets

| Dataset | 현재 판단 | 이유 |
|---|---|---|
| RE10K | main benchmark 1순위 후보 | MVSplat 공식 pretrained/evaluation이 RE10K 중심. MVSplat을 공정하게 넣기 좋음 |
| DL3DV | main benchmark 강한 후보 | DepthSplat이 DL3DV 및 RE10K+DL3DV checkpoint/실행 예시를 제공. 다운로드는 Hugging Face access와 용량 계획 필요 |
| DTU | external validation / C2 | GT geometry가 있어 depth, Chamfer, floater, free-space opacity 분석에 좋지만 FF에는 OOD penalty가 큼 |
| ACID | secondary 후보 | MVSplat이 ACID pretrained도 제공하지만 프로젝트 주제/로봇 앵커와의 직접성은 RE10K/DL3DV보다 약함 |

## Immediate decision

1. MVSplat을 반드시 main FF baseline으로 유지한다면 RE10K가 가장 자연스럽다.
2. DepthSplat을 핵심 FF baseline으로 끌어올리고 더 복잡한 실제 장면/로봇 앵커를 원하면 DL3DV를 적극 검토한다.
3. 둘 다 쓰려면 main benchmark를 하나로 고르기 전에 `RE10K small subset`과 `DL3DV small subset`의 확보 난이도와 checkpoint compatibility를 직접 확인한다.


## Resolution policy

해상도는 우리가 임의로 고르는 축이 아니다. Feed-forward checkpoint가 학습된 해상도를 상속한다.

- RE10K main candidate: pixelSplat/MVSplat/DepthSplat 표준에 맞춰 `256x256` chunk를 사용한다.
- DL3DV DepthSplat candidate: DL3DV checkpoint는 주로 `256x448` 계열을 사용한다.
- Per-scene optimization baseline도 main comparison에서는 반드시 같은 입력 이미지 해상도와 crop을 사용한다. RE10K 본 실험에서 Vanilla3DGS만 원본 고해상도를 쓰면 비교가 무효다.
- DTU 원본 `1600x1200` run은 dense sanity/external geometry 검증용이며, RE10K main protocol과 섞지 않는다.

## Local data status

RE10K small subset 확보 완료:

- Source: pixelSplat official small subset Google Drive folder.
- Raw zip: `/data/Re-feem/raw_downloads/pixelsplat_small/re10k/re10k_subset.zip`
- SHA256: `fe08f7aad99d0fe9c171aa585816bb5f372a5e8804ac330a6dfb01e98fec8809`
- Extracted root: `/data/Re-feem/datasets/re10k`
- Train split: 39 scenes, 3 chunks.
- Test split: 41 scenes, 3 chunks.
- Zip integrity: `unzip -t` passed.

주의: `.torch` chunk는 pickle 기반이므로 자동 검증 단계에서는 `torch.load`로 역직렬화하지 않았다. 실제 runner에서 공식 데이터 로드를 승인한 뒤 해당 모델 env에서 읽는다.

## Data acquisition plan

### RE10K

- 목적: MVSplat in-domain main benchmark 후보.
- 확인할 것: pixelSplat/MVSplat preprocessed chunk 접근성, 삭제된 YouTube 영상 비율, pose/split 재현 가능성.
- 저장 위치: `/data/Re-feem/datasets/re10k`.

### DL3DV

- 목적: DepthSplat in-domain 또는 near-domain main benchmark 후보.
- 확인할 것: Hugging Face access 승인, 480P/960P images+poses subset 다운로드, COLMAP cache 필요 여부.
- 저장 위치: `/data/Re-feem/datasets/dl3dv`.

### DTU

- 목적: external geometry validation, dense-view sanity check, C2.
- 현재 상태: 공식 sparse-view split 16개 모두 확보 완료.
- 공식 split: `1,8,21,30,31,34,38,40,41,45,55,63,82,103,110,114`.

## Next action

main dataset을 확정하기 전, 아래 둘을 먼저 끝낸다.

1. DTU dense-view 49-view sanity check로 Vanilla 3DGS pipeline이 정상 PSNR 범위에 도달하는지 확인.
2. DepthSplat/MVSplat model zoo와 checkpoint별 training dataset/view-count 지원 표를 채운다.
