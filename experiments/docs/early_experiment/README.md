# Early Experiment Setup

이 디렉터리는 본 실험 전에 동결해야 하는 STEP 0~2 내용을 따로 모아둔다. 목표는 모델 성능을 빨리 보는 것이 아니라, regime map의 축과 선택 규칙을 먼저 고정하는 것이다.

## STEP 0. Visibility와 overlap 산출

입력은 SfM visibility다. COLMAP text export의 `images.txt` 또는 `{view_id: [point_id, ...]}` 형태의 JSON을 사용한다.

```bash
python3 experiments/scripts/generate_overlap.py \
  --colmap-images /data/Re-feem/datasets/re10k/scene_000/sparse/0/images.txt \
  --output-dir experiments/outputs/overlap/re10k_scene_000/4view_seed0 \
  --scene re10k_scene_000 \
  --view-count-label 4view_seed0
```

특정 view subset만 평가하려면 newline-delimited view 목록을 넘긴다.

```bash
python3 experiments/scripts/generate_overlap.py \
  --visibility-json /data/Re-feem/datasets/re10k/scene_000/visibility.json \
  --views experiments/docs/early_experiment/example_views.txt \
  --output-dir experiments/outputs/overlap/re10k_scene_000/manual_subset
```

## 출력 파일

- `summary.json`: mean overlap, q25, median reference, zero-pair ratio, isolated-view flag
- `pairwise_overlap.csv`: 모든 view pair의 shared point 수와 overlap

`mean_overlap`이 주 지표다. 매칭 실패 pair와 공통 point가 부족한 pair는 제외하지 않고 overlap 0으로 들어간다.

## STEP 1. 파일럿 5장면

파일럿에서는 RE10K 5장면을 골라 view 수 `2/4/8/12`, seed `0/1/2`에 대해 overlap 분포를 먼저 만든다. 각 view 수 안에서 low/high threshold를 층화해 정하고, 전체 분포 하나로 공통 threshold를 만들지 않는다.

필수 확인 항목:

- view 수별 overlap histogram
- zero-pair ratio 분포
- isolated-view sample 목록
- texture covariate 준비 상태
- 모델별 지원 view 수 smoke test 결과

## STEP 2. 본 실험 전 동결

아래 값은 결과를 보기 전에 고정한다.

- low/high overlap threshold per view count
- PSNR tau: `max(pilot seed variability, 0.5dB)`
- budget checkpoint rule: 예산 종료 직전 checkpoint
- oracle peak 저장 경로: `experiments/outputs/oracle_results`
- C1-b renderer equivalence tolerance: `0.0001`
- C2 representative conditions: `2view_low_overlap`, `4view_low_overlap`, `4view_high_overlap`, `12view_high_overlap`

이 문서의 값이 config와 다르면 config를 우선 수정하고 manifest를 다시 생성한다.
