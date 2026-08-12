# 일일 보고서 — 2026-08-11

## 오늘 목표

전날(8/10) 마무리한 스크립트 재구성(core/runners/probes/batch/analysis) 위에서, main benchmark 결정 근거를 문서로 굳히고 2/4/8/12-view 통제 축이 실제로 안 깨지는지 DTU에서 스모크 테스트로 확인하는 것. docs/ 디렉토리도 scripts/와 같은 방식으로 유형별로 정리.

---

## 1. 문서 정리

README.md, data_acquisition_plan.md, implementation_summary.md, model_checkpoint_domain_table.md 네 개를 갱신해 지금 상태(RE10K 114 scene, DL3DV 25 scene, DTU 29 scan 확보 완료)를 반영. 결론: "데이터부터 받아야 함" 단계는 지났고, main benchmark를 RE10K-first로 잠그는 단계로 넘어감.

- MVSplat이 RE10K in-domain이고 114 scene을 이미 확보했다는 근거로 RE10K를 1순위 후보로 판단.
- DL3DV는 DepthSplat 중심 secondary benchmark로 위치 정리 (MVSplat을 DL3DV에 쓰면 OOD caveat 필요).
- DTU는 external validation / C2 / dense-sanity 전용이라는 기존 원칙 재확인.
- 해상도는 임의 선택이 아니라 feed-forward checkpoint가 학습된 해상도를 상속(RE10K 256x256, DL3DV 256x448)해야 한다는 원칙 명문화.

## 2. DTU scan1 2/4/8/12-view 통제 스모크 테스트

experiments/scripts/analysis/generate_dtu_view_overlap_smoke.py 신규 작성 — runner와 동일한 seed 기반 view 선택 규칙을 재사용해 view set을 만들고, COLMAP known-pose triangulation -> overlap report -> Vanilla3DGS(10초 budget)/MVSplat 실행까지 한 번에 검증.

| View | Selected views | SfM points | Mean overlap | Zero-pair ratio | Vanilla3DGS(10s) | MVSplat |
|---|---|---:|---:|---:|---|---|
| 2 | 32, 41 | 0 | 0.000 | 1.000 | OK (random-init fallback) | OK |
| 4 | 14, 25, 31, 40 | 313 | 0.517 | 0.000 | OK | OK |
| 8 | 2, 3, 5, 13, 16, 23, 27, 35 | 1,994 | 0.416 | 0.000 | OK | OK |
| 12 | 2, 3, 4, 9, 12, 13, 20, 25, 32, 39, 46, 48 | 3,327 | 0.247 | 0.000 | OK | OK |

2-view는 다시 SfM point 0개(overlap 0.000, isolated) — 어제(8/10) scan30에서 발견해 고친 COLMAP 0-match 버그가 여기서도 발동했는데, 이번엔 fatal crash가 아니라 init_source: random_sphere_fallback으로 정확히 넘어간 것까지 로그로 확인(직접 검증함, 아래 3절).

주의(원 문서에도 명시돼 있던 caveat, 그대로 유지): 이건 DTU external track용 배관 검증이다. RE10K main benchmark용 view candidate/overlap bucket은 아직 없음. MVSplat 4/8/12-view는 "forward pass가 죽지 않는다"는 뜻이지 학습 분포 내 공식 지원이 확정됐다는 뜻이 아니다(5.2절 표는 여전히 미확정).

## 3. 검증 (직접 재확인함)

다른 세션이 요약해둔 결과를 코드/로그로 직접 대조했다 — 전부 실제 결과와 일치.

- python3 -m unittest discover -s tests -> 9개 통과 (재확인)
- bash experiments/scripts/batch/run_experiment.sh -> manifest 12,480 row 정상 생성 (재확인)
- experiments/outputs_smoke_20260811/overlap/dtu_scan1/{2,4,8,12}view_seed0/summary.json의 mean_overlap 값이 위 표와 정확히 일치 (0.000/0.517/0.416/0.247)
- experiments/outputs_smoke_20260811/logs/dtu_scan1_Vanilla3DGS_2view_seed0.json 마지막 체크포인트에서 init_source: "random_sphere_fallback" 직접 확인
- batch_summary.json의 8개 run(Vanilla3DGS/MVSplat x 2/4/8/12-view) 전부 status: "ok", elapsed 시간도 타당한 범위(17~19s / 6~8s)

## 4. experiments/docs/도 scripts/와 같은 방식으로 재구성

```
daily_logs/   daily_report_*.md, critical_path_2026-08-10.md, data_acquisition_plan_2026-08-10.md,
              paper_scaffold_audit_log.md  (시간순 기록 - "paper_" 이름이지만 내용은 로그)
reference/    dataset_recommendation.md, model_checkpoint_domain_table.md, implementation_summary.md
paper/        paper_gauss_newton_notation.md, paper_priority_list.md, paper_reading_log.md,
              paper_reading_template.md  (논문에 실제로 들어갈 내용)
early_experiment/  (기존 그대로)
```

문서 간 상호 참조는 전부 prose 언급이라 실제 markdown 링크 깨짐 없음(확인함). current_status_2026-08-11.md는 이 보고서로 흡수하고 삭제.

---

## 종합 결론

1. 어제(8/10) "파이프라인이 도는가"에 이어, 오늘은 "view 축(2/4/8/12)이 안 깨지는가"까지 DTU에서 확인됐다. 다른 세션의 작업 내용을 전부 코드/로그로 직접 재검증했고 전부 사실과 일치했다.
2. 2-view 케이스가 다시 한번 0-overlap fallback을 발동시켰다 — 버그가 아니라 sparse-view가 실제로 이런 조건을 만들어낸다는 것 자체가 이 연구의 핵심 현상이라는 걸 재확인시켜준 사례.
3. 본 실험 병목이 "데이터"에서 "RE10K에 이 스모크와 같은 축을 그대로 적용하는 일"로 완전히 이동했다. DTU에서 검증된 view-selection/overlap/runner 삼각형이 아직 RE10K에는 안 붙어 있음.
4. 문서 구조가 코드 구조와 같은 패턴으로 정리됐다 — scripts/는 8/10에, docs/는 오늘.

---

## 확인된 진행상황 체크리스트

### 연구 설계 / 프로토콜
- [x] 연구 방향 문서화, 실험 config 작성
- [x] budget checkpoint / oracle checkpoint 분리
- [x] scene 단위 cluster bootstrap 규칙
- [x] overlap non-edge=0 포함 규칙 구현
- [x] C1-b / C2 통제 실험 축 설계
- [x] dense-view sanity 필요성 반영 및 실행
- [x] main benchmark 해상도 = checkpoint 상속 원칙 확정
- [ ] 5.2절 모델별 지원 view 수 / confidence 출력 표 완성
- [ ] 5.4절 GPU-hour 예산 재계산

### 데이터
- [x] /data/Re-feem 디렉토리 구성
- [x] DTU 공식 split 16 scan + extra 13 scan = 29 scan
- [x] RE10K test 114 scene (probe 41 + mirror 73)
- [x] DL3DV pilot 25 scene, 공식 benchmark split과 중복 0 확인
- [x] RE10K/DL3DV SOURCE.md 작성
- [ ] RE10K main subset(20~30 scene) index 생성
- [ ] RE10K citation/license 문구 논문용 정리

### 모델 / 러너
- [x] Vanilla3DGS runner (COLMAP init, LPIPS, densification 궤적 로깅)
- [x] MVSplat 정식 runner (protocol_utils 스키마)
- [x] DepthSplat probe (정식 runner 아님)
- [x] model_registry 구조 + batch driver 일반화
- [x] DTU 공식 split 16 scan x 2-view x seed0 batch
- [x] DTU scan1 2/4/8/12-view 통제 스모크 (오늘)
- [ ] DepthSplat probe -> 정식 runner 승격
- [ ] SparseGS/FSGS 통합 (미착수)
- [ ] RE10K .torch chunk를 정식 runner 입력으로 연결
- [ ] densification on/off CLI 추가
- [ ] co-visibility 기반 view selector - runner와 generate_overlap.py 연결

### 검증 결과
- [x] unit test 9개 통과
- [x] manifest 12,480 row 생성
- [x] DTU dense-view sanity (scan1, 42-view, 30k iter, 24dB대)
- [x] MVSplat RE10K in-domain (25.6dB / 22.4dB)
- [x] MVSplat DTU zero-shot (4.8~11.8dB, OOD로 해석)
- [x] DepthSplat DL3DV in-domain (20.0dB)
- [x] DTU 2/4/8/12-view 통제 스모크 (오늘)
- [ ] RE10K에서의 동일한 2/4/8/12-view 통제 스모크
- [ ] RE10K/DL3DV용 overlap bucket (low/high threshold) 산출

### 분석 / 이론
- [x] Gauss-Newton / J^T J 수식 정리 문서
- [x] geometry uncertainty figure 1차 생성 + 실측
- [x] overlap-uncertainty 부호 이상 현상 발견 및 원인 분석(baseline confound) - 재해석은 A-1 완료 후로 남음
- [ ] co-visibility 기반 view selector 구현 (도구는 있음, runner와 연결이 안 됨)
- [ ] overlap bucket threshold를 본 실험용으로 동결

대략적 진행률 감각(검증 결과 기준으로 재확인): 연구 설계 ~80%, 데이터 확보 ~80%, 스크립트 scaffold ~75%, 모델별 정식 runner ~55%, 본 실험 자동 실행 라인 ~40~50%. 논문에 쓸 결과 생산 이전, 파일럿 직전 단계.

---

## 다음 실행 목록 (8/12로 이월)

1. RE10K 20~30 scene main subset index 생성 (114개 중 부분집합 선정 방식 결정 필요)
2. RE10K .torch chunk loader를 runners/ 러너의 공통 입력으로 정리
3. RE10K 2/4/8/12-view candidate + overlap bucket 생성 - 오늘 만든 스모크 패턴을 RE10K용으로 이식
4. Vanilla3DGS runner에 RE10K 256x256 input path 연결
5. DepthSplat 정식 runner 작성 (depthsplat_runner.py, protocol_utils 스키마)
6. co-visibility 기반 view selector를 runner에 실제로 연결
7. densification on/off CLI 추가
8. A-1(Gauss-Newton) 완료 후 geometry uncertainty 재해석
9. RE10K SOURCE.md 인용/라이선스 문구 정리
10. 5.2절 표, 5.4절 GPU-hour 재계산
