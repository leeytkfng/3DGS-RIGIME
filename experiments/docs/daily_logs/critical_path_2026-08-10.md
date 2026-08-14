# Critical Path Update — 2026-08-10

## 정정

직전 평가에서 "스모크 테스트 0회"처럼 표현했던 것은 틀렸다. 현재 상태는 단순 스캐폴드가 아니라, 1주차에 기대하던 상당 부분을 하루 안에 밀어붙인 상태다. Vanilla 3DGS와 MVSplat 두 계열이 모두 실데이터(DTU)에서 실행됐고, 계획서가 사전에 고정한 주요 프로토콜 규칙이 실제 러너와 로그에서 깨지지 않는 것도 확인됐다.

## 특히 가치 있었던 확인

1. `remotezip` range read로 DTU 전체 압축 파일을 받지 않고 필요한 scan만 확보했다. 130GB급 전체 파일 대신 scan 단위 데이터만 가져온 것은 이후 데이터 확보 전략에도 유효하다.
2. COLMAP은 pose estimation이 아니라 known-pose triangulation으로만 사용했다. 학습 input view만 넣고 held-out test view와 GT point cloud를 쓰지 않아 leakage를 막았다.
3. H200 단일 GPU에서 여러 3DGS run을 동시에 돌려도 총 처리량이 늘지 않는다는 것을 실측했다. 부정적 결과지만 GPU-hour 계획을 직접 바꾸는 값이라 중요하다.

## 위험 1. Vanilla 3DGS 파일럿 PSNR이 너무 낮다

DTU scan1 8-view에서 60초 PSNR 10.69dB, 300초 PSNR 9.85dB는 정식 성능 해석에 사용할 수 없다. random init과 짧은 iteration 때문에 낮아질 수는 있지만, 10dB대는 배경 처리, 마스킹, 노출 정규화, pose/rasterization convention 중 하나가 어긋난 상태에서도 나올 수 있다.

특히 `Gaussian count 증가 + test PSNR 하락`은 계획서 H1/H2와 정성적으로 닮아 보이므로 더 위험하다. 망가진 파이프라인에서도 같은 모양의 곡선이 나올 수 있다. 따라서 이 관측은 가설 지지 증거가 아니라, 현재까지는 로깅 배관이 신호를 기록할 수 있다는 정도로만 취급한다.

### 즉시 필요한 sanity check

DTU 한 scan을 dense view(49장 전체) + 표준 iteration(30k) 또는 그에 준하는 충분한 iteration으로 돌려 알려진 정상 범위가 나오는지 확인한다.

- 기대: DTU dense-view setting에서 25dB 이상 수준의 정상값에 접근해야 한다.
- 정상값이 나오면: pose/rasterization/evaluation pipeline은 대체로 맞고 sparse 저하는 진짜 현상으로 볼 수 있다.
- 정상값이 안 나오면: sparse 실험 스케일업을 멈추고 배경/마스크/노출/좌표계부터 재점검한다.

이 check 없이 scan/seed를 늘리면 전부 다시 돌릴 위험이 있다.

## 위험 2. DTU는 main benchmark가 아니라 external/C2 track이다

MVSplat의 DTU zero-shot PSNR 4.8~11.8dB는 RE10K-trained feed-forward model이 DTU에서 강한 OOD penalty를 받는다는 신호다. Per-scene optimization은 장면마다 새로 맞추므로 같은 의미의 학습 도메인 shift가 없다. 따라서 DTU에서 FF vs optimization을 main comparison으로 밀면, 패러다임 비교가 아니라 `OOD penalty를 맞은 FF vs scene-specific optimization` 비교가 된다.

결론: 계획서 원칙대로 DTU는 external geometry validation과 C2/failure analysis에 둔다. Main dataset은 feed-forward 공개 checkpoint의 학습 도메인과 맞는 쪽으로 정해야 한다.

## 주 데이터셋 결정 기준

주 데이터셋은 다운로드 편의보다 아래 기준을 먼저 본다.

1. Feed-forward checkpoint가 해당 dataset 또는 가까운 domain으로 공개돼 있는가?
2. Pose-given evaluation protocol과 train/test view split을 재현할 수 있는가?
3. View count 2/4/8/12 또는 그 축소판을 모델별로 공정하게 지원할 수 있는가?
4. 데이터 접근성이 현실적인가?

## 현재 공식 자료 기준 판단

- MVSplat 공식 repo는 RealEstate10K/ACID pretrained evaluation을 제공하고, DTU는 RE10K checkpoint의 cross-dataset generalization evaluation으로 둔다. 따라서 MVSplat을 공정하게 main comparison에 넣으려면 RE10K가 가장 자연스럽다.
- DepthSplat 공식 repo는 RE10K용 checkpoint뿐 아니라 DL3DV 및 RE10K+DL3DV 계열 checkpoint와 2/6/12-view 실행 예시를 제공한다. 따라서 DepthSplat 관점에서는 DL3DV도 strong candidate다.
- DL3DV-10K는 Hugging Face를 통해 480P/960P images+poses와 COLMAP cache 등을 제공하지만 access request와 용량 계획이 필요하다.

## Split 결정

DTU는 공식 MVSplat `convert_dtu.py`에 박힌 sparse-view test split으로 갈아탄다. 현재 `/data/Re-feem/datasets/dtu`에는 공식 16개 scan이 모두 존재한다.

공식 split:

`1, 8, 21, 30, 31, 34, 38, 40, 41, 45, 55, 63, 82, 103, 110, 114`

DTU는 external/C2용이므로 이 16개 전체를 항상 쓸 필요는 없다. 파일럿은 6~10개 subset으로 충분하고, 최종 external validation에서 8~15개를 선택한다.

## 재정렬된 우선순위

1. **Main dataset 결정**: RE10K vs DL3DV. 반드시 DepthSplat/MVSplat 공개 checkpoint 도메인과 함께 결정한다.
2. **Dense-view sanity check**: DTU 49-view + 충분한 iteration으로 Vanilla 3DGS 기준값 확보.
3. **Model support table (§5.2)**: MVSplat 4/8/12-view 동작 여부와 confidence/uncertainty output 유무 확인. DepthSplat도 checkpoint별 지원 view 수를 채운다.
4. **DTU official split 전환**: 이미 데이터는 모두 있으므로 config/docs/log naming을 공식 split 기준으로 정리한다.
5. **GPU-hour 재계산**: C1-b/C2 포함 2.4배 증가 + 단일 GPU 병렬화 무효를 반영한다.
6. **Runner/driver/overlap 스케일업**: 위 sanity check 후에 scan/seed batch를 늘린다.
7. **RE10K/DL3DV 데이터 확보**: main dataset 결정 후 최소 파일럿 subset부터 받는다.

## 현재 결론

프로젝트는 예상보다 앞서 있다. 다만 지금 당장 스케일업하면 위험한 지점이 분명하다. 다음 작업은 더 많은 sparse run이 아니라, main dataset 결정과 dense-view sanity check다.


## 2026-08-10 RE10K probe 확보

피드백 반영 후 pixelSplat 공식 small subset에서 `re10k_subset.zip`만 직접 다운로드했다. 전체 folder download는 ACID부터 받기 시작해 중단했고, RE10K file id만 지정해 다시 받았다.

결과:

- `/data/Re-feem/datasets/re10k/test`: 41 scenes, 3 `.torch` chunks.
- `/data/Re-feem/datasets/re10k/train`: 39 scenes, 3 `.torch` chunks.
- 이 subset은 전체 500GB train+test가 아니라 small/probe subset이다.
- 계획서 규모 20~30 scenes의 파일럿에는 test split만으로 충분하다.

다음 검증:

1. 공식 reader 또는 MVSplat/pixelSplat dataset loader로 `.torch` chunk를 안전하게 로드한다.
2. scene id와 frame count, image shape가 `256x256`인지 확인한다.
3. RE10K main comparison에서는 Vanilla3DGS도 같은 256x256 image/crop으로 학습·평가하도록 runner를 분리한다.
