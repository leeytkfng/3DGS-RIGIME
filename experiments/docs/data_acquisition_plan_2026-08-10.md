# Data acquisition plan — 2026-08-10

## 현재 로컬 상태

- `/data/Re-feem/datasets/dtu`: DTU 공식 split 16개 + extra scan 확보됨.
- `/data/Re-feem/datasets/re10k`: 비어 있음.
- `/data/Re-feem/datasets/dl3dv`: 비어 있음.
- `/data/Re-feem/code/mvsplat`: 공식 repo와 RE10K checkpoint 확보됨.

## 즉시 받을 데이터 우선순위

### 1. RE10K small/probe subset

목적: MVSplat in-domain main benchmark 가능성 확인.

해야 할 일:

1. pixelSplat/MVSplat preprocessed chunk 접근 가능 여부 확인.
2. 삭제된 YouTube 비율이 문제가 되는 raw download 대신 processed subset을 우선 시도.
3. 최소 5 scene test subset만 먼저 확보.
4. 저장 위치: `/data/Re-feem/datasets/re10k`.

### 2. DL3DV small/probe subset

목적: DepthSplat in-domain main benchmark 가능성 확인.

해야 할 일:

1. Hugging Face dataset access request 여부 확인.
2. 우선 480P 또는 DepthSplat preprocessed subset부터 시도.
3. 최소 5 scene images+poses 또는 `.torch` chunk 확보.
4. 저장 위치: `/data/Re-feem/datasets/dl3dv`.

### 3. DTU는 추가 다운로드 중단

현재 공식 split 16개가 모두 있으므로 DTU 추가 다운로드보다 RE10K/DL3DV 확보가 우선이다.

## 주의

- Hugging Face gated dataset은 CLI token/auth가 필요할 수 있다.
- 대용량 dataset을 바로 받지 말고 probe subset부터 받는다.
- dataset license/access 기록을 docs에 남긴다.
