# Data Acquisition Status — 2026-08-10 / updated 2026-08-11

이 문서는 원래 "무엇을 먼저 받을지"를 정하는 계획서였지만, 2026-08-10 작업 후에는 데이터 확보 병목이 상당 부분 해소됐다. 지금부터는 다운로드 계획보다 **어떤 데이터셋을 main benchmark로 둘지**와 **각 feed-forward checkpoint의 in-domain 조건을 맞추는지**가 핵심이다.

## 현재 로컬 상태

- `/data/Re-feem/datasets/re10k`: RE10K test 중심 `.torch` chunk 확보. `test/000000.torch`~`000007.torch`, 총 114 scene, 약 1.2GB. 기존 pixelSplat small subset 41 scene에 Hugging Face mirror 73 scene을 추가했고 중복은 0으로 확인했다. `SOURCE.md`에 출처와 주의사항 기록 완료.
- `/data/Re-feem/datasets/dl3dv`: DL3DV pilot 25 scene 확보. 각 scene은 `transforms.json`과 `images_8`를 갖고 있으며 구조 검증 완료. 공식 `DL3DV-Benchmark` 140 scene과 중복 0으로 확인했다.
- `/data/Re-feem/datasets/dtu`: DTU 공식 sparse-view split 16개 scan을 포함해 총 29 scan 보유. DTU는 main이 아니라 external validation, C2, dense sanity 용도다.
- `/data/Re-feem/code/mvsplat`: 공식 MVSplat repo와 RE10K checkpoint 확보. RE10K in-domain probe 정상.
- `/data/Re-feem/code/depthsplat`: 공식 DepthSplat repo, DL3DV checkpoint, 테스트 subset 확보. DL3DV in-domain probe 정상.

## 검증 완료 내용

### RE10K

목적: MVSplat in-domain main benchmark 가능성 확인.

- pixelSplat/MVSplat 계열과 같은 `.torch` chunk 형식으로 확보했다.
- 256x256 protocol을 상속해야 한다. feed-forward checkpoint가 이 해상도로 학습됐으므로 per-scene 3DGS도 같은 입력 해상도로 학습·평가해야 한다.
- MVSplat 공식 RE10K checkpoint probe 결과가 정상 범위였다. 기존 small subset은 mean PSNR 25.6dB, 추가 mirror chunk도 22.4dB로 확인됐다.
- 현재 114 scene은 계획서의 main 규모 20~30 scene보다 충분하다.

### DL3DV

목적: DepthSplat in-domain main benchmark 가능성 확인.

- 25 scene 확보, 구조 검증 완료.
- DepthSplat DL3DV checkpoint(`256x448`, random view 2~6 계열)로 probe 완료.
- context view를 너무 넓게 잡으면 약 12dB까지 떨어지고, 가까운 context에서는 mean PSNR 20.0dB 수준으로 개선됐다. 따라서 DL3DV runner는 임의 프레임 간격이 아니라 co-visibility/near-context 기반 view 선택이 필요하다.
- 공식 benchmark split과 중복 0이라 train-pool 성격의 pilot으로 안전하다.

### DTU

목적: external validation / C2 / dense sanity.

- 공식 split 16개는 모두 확보되어 있다: `1, 8, 21, 30, 31, 34, 38, 40, 41, 45, 55, 63, 82, 103, 110, 114`.
- 추가 자체 선정 scan도 남아 있어 총 29 scan이다.
- Vanilla 3DGS dense-view sanity는 scan1, 42 train view, 30k iteration에서 PSNR 24.0~24.1dB, SSIM 0.843, LPIPS 0.218 수준으로 정상 배관을 확인했다.
- MVSplat의 DTU zero-shot 저PSNR은 in-domain 문제가 아니라 OOD penalty로 해석하는 것이 맞다. 따라서 DTU를 FF vs per-scene main 비교로 쓰면 공정성 교란이 생긴다.

## 현재 결정해야 할 것

1. **Main benchmark 선택**: RE10K-first가 현재 가장 공정하다. MVSplat이 in-domain이고 데이터도 114 scene 확보되어 있다. DepthSplat도 RE10K checkpoint가 있으므로 둘을 같은 도메인에서 비교할 여지가 있다.
2. **DL3DV의 역할**: DepthSplat 중심 보조 main 또는 second benchmark 후보. 단 MVSplat은 DL3DV in-domain checkpoint가 없으면 OOD 비교가 될 수 있다.
3. **해상도 고정**: RE10K main은 256x256, DepthSplat DL3DV track은 해당 checkpoint의 256x448을 따라야 한다. per-scene 3DGS도 각 track의 feed-forward 입력 해상도를 그대로 상속한다.
4. **View selection**: random/간격 기반 선택은 이제 위험하다. `generate_overlap.py`의 co-visibility 계산과 runner를 연결해 overlap bucket을 명시적으로 구성해야 한다.

## 다음 실행 순서

1. RE10K를 main pilot으로 확정할지 결정하고, 20~30 scene subset index를 만든다.
2. MVSplat/DepthSplat의 RE10K checkpoint와 지원 view 수(2/4/8/12 또는 checkpoint별 범위)를 `model_checkpoint_domain_table.md`에 확정 반영한다.
3. Vanilla 3DGS runner에 256x256 resize/crop protocol을 명시하고 RE10K chunk loader를 붙인다.
4. DepthSplat probe를 `depthsplat_runner.py` 정식 runner로 승격한다.
5. co-visibility 기반 view selector를 공통 모듈로 만들고 RE10K/DL3DV/DTU runner가 같은 선택 로직을 쓰게 한다.
6. RE10K `SOURCE.md`의 mirror 출처, license, citation 표기를 논문/README용으로 정리한다.

## 보류 또는 낮은 우선순위

- DTU 추가 다운로드: 이미 충분하므로 중단.
- 전체 RE10K 543 chunk 다운로드: 현재 114 scene으로 파일럿/본 실험 규모를 넘기므로 필요할 때만 확장.
- DL3DV full benchmark 다운로드: gate 승인과 저장공간 계획이 필요하다. 현재는 25 scene pilot으로 충분하다.
