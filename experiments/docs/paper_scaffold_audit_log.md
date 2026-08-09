# Scaffold audit log — 2026-08-09

이 문서는 실험 스캐폴드(config/scripts/tests)를 실행해서 검증한 결과와, 그 결과를 계획서(§5.4, §5.7, §5.12) 및 담당자 검토와 대조해 확정한 사항을 기록한다. 목적은 코드를 더 다듬기 전에 "설계와 구현이 일치하는가", "GPU 예산 추정이 여전히 유효한가"를 문서로 동결하는 것이다.

## 1. 실행 검증 결과

- `python3 -m unittest discover -s tests` → 9개 테스트 전부 통과
- `bash experiments/scripts/run_experiment.sh` → `experiment_manifest.json` 생성, 총 12,480 row

## 2. Manifest 구조 재확인 (담당자 검토 반영)

담당자가 12,480을 26장면 × 5method로 역산했으나, manifest를 직접 파싱한 결과와 다르다. 실제 구조는 아래처럼 **3개 phase의 합**이며 단일 6중 product가 아니다.

| phase | rows | 구성 |
|---|---|---|
| main | 7,680 | scenes(20) × seeds(3) × view_counts(4) × overlap(2) × budgets(4) × methods(4) |
| c1b | 3,840 | scenes(20) × seeds(3) × view_counts(4) × overlap(2) × ff_methods(2) × (off 1 + on budgets 3) |
| c2_depth_noise | 480 | external_scenes(8) × seeds(3) × representative_conditions(4) × noise_levels(5) |
| c2_depth_scale_bias | 480 | external_scenes(8) × seeds(3) × representative_conditions(4) × scale_levels(5) |
| **합계** | **12,480** | |

- **method는 정확히 4개**: `DepthSplat`, `MVSplat`, `Vanilla3DGS`, `SparseGS` (manifest에서 직접 확인). 5번째 method는 없다.
- **C1-b는 method 목록에 섞여 있지 않다.** `phase: "c1b"`로 완전히 분리된 row이고, 각 row에 `renderer_equivalence_gate: true`가 별도로 기록된다. 담당자가 우려한 "gate 무력화" 시나리오는 현재 구조에서는 발생하지 않는다.
- main phase의 scene 수는 20 (`scenes_primary`), 외부 검증(C2)은 8 (`scenes_external`)이며 두 값이 다른 숫자다. 26이라는 값은 어디에도 대응하지 않는다.

## 3. Optimization 실행 횟수 재계산 — §5.4 GPU-hour 추정과의 정합성

budget(1/10/60/300초)은 **하나의 연속 학습 궤적에서 뽑는 체크포인트**이지 재학습이 아니다. 이 전제로 실제 optimization 실행(학습을 새로 시작하는 단위) 수를 다시 세면:

- main phase, optimization 계열만: `scenes(20) × seeds(3) × view_counts(4) × overlap(2) × opt_methods(2)` = **960회**
- 여기에 C1-b `refinement=on` 트랙(FF 초기값에서 새로 시작하는 refinement 실행)을 더하면: `20 × 3 × 4 × 2 × ff_methods(2)` = **960회 추가**
- C2는 noise/scale 조건마다 초기화 자체가 달라서 체크포인트 공유가 불가능하므로 전부 별도 실행: `480 + 480` = **960회**

**합계 실제 optimization 실행 ≈ 2,880회**로, §5.4의 최초 추정 "약 1,200 optimization run"보다 약 2.4배 많다. main phase만 보면 960회로 최초 추정과 비슷하지만, C1-b와 C2를 포함한 총량은 재검토가 필요하다.

**결론: §5.4의 GPU-hour 추정(200~300 GPU-hour)은 main phase 단독 기준으로는 대략 맞고, C1-b·C2를 포함한 전체 기준으로는 과소추정일 가능성이 높다. 파일럿 실측 후 재계산 필요 (계획서 §12 STEP 2에서 이미 요구하는 절차와 동일).**

## 4. 확인된 placeholder / 미확정 값 (실제 값으로 갱신 전 사용 금지)

- `model_registry.py`의 네 모델 모두 `supports_views=[2, 4, 8, 12]`로 동일 — 스모크 테스트 전 임시값, §5.2 표를 대신할 수 없음
- `experiment_config.yaml`의 `analysis.tau.pilot_seed_variability_psnr: 0.0` — 파일럿 seed 변동성 실측 전 placeholder. 현재 tau는 사실상 `practical_min_delta_psnr=0.5`로만 고정됨 (보수적 방향이라 안전하지만 갱신 필요)
- `run_experiment.py`의 scene id(`re10k_scene_000` 등)는 실제 데이터셋 스캔 결과가 아니라 deterministic placeholder

## 5. 환경 확인 (2026-08-09)

- GPU: NVIDIA H200 NVL, 143,771 MiB, idle (0% util, 사용 중인 프로세스 없음)
- 외부 네트워크 접근: 가능 (`curl` 테스트 성공)
- `/data/Re-feem/datasets/{re10k,dtu,dl3dv}`, `sfm_exports`, `overlap_reports`, `raw_downloads` 전부 빈 디렉터리 — 데이터 미확보 상태

## 6. 다음 결정이 필요한 항목 (블로킹)

1. DTU 데이터 배포본 선택 — 원본 DTU MVS(대용량, GT point cloud 포함, C2 depth AbsRel/RMSE·Chamfer 계산에 필요) vs 커뮤니티에서 흔히 쓰는 전처리된 sparse-view subset. 소스 URL 확정 필요.
2. 실제 모델 러너 통합 우선순위 — DepthSplat/MVSplat/Vanilla3DGS/SparseGS 중 스모크 테스트를 먼저 통과시킬 1개 선정. GPU 메모리 활용(batch size, resolution 등) 코드 변경은 이 통합이 선행돼야 의미가 있음.
