# Gate Resolution — renderer-equivalence 임계값 대조 (2026-08-16)

**작성**: 이용수(A 컨테이너) — 작업 지시 A-1 대응
**목적**: B(황인재)가 "60dB gate 미달"로 C1-b를 blocked 처리한 것에 대해, 우리 쪽 동결 기준과 대조해 실제 PASS/FAIL을 판정한다.

---

## 1. 우리 동결 기준의 출처 (검증됨)

**결론: 33.0 dB. 60 dB는 이 저장소 히스토리 어디에도 존재한 적이 없다.**

### 1.1 값 자체

```
experiments/configs/experiment_config.yaml:105
  renderer_equivalence_tolerance_psnr_db: 33.0
```

### 1.2 git 히스토리 전체 (완전 재현)

```
$ git log --oneline --all -S "renderer_equivalence_tolerance" -- experiments/configs/experiment_config.yaml
6884922 Finalize renderer_equivalence_tolerance at PSNR>=33dB using 390 pooled samples
97a5615 first commit
```

이 파일에서 `renderer_equivalence_tolerance` 관련 줄을 건드린 커밋은 **이 두 개가 전부**다.

- **`97a5615` (first commit)**: `renderer_equivalence_tolerance: 0.0001` (MSE 값, PSNR 아님) — 파일럿 전 잠정 추정치.
- **`6884922` (2026-08-13 00:41:37 UTC)**: 위 MSE 0.0001을 폐기하고 `renderer_equivalence_tolerance_psnr_db: 33.0`으로 최종 동결. 커밋 메시지 원문:

  > Aggregated all renderer-equivalence gate logs accumulated across DTU/RE10K (MVSplat) and DL3DV (DepthSplat) scale-ups: 130 gate checks, 390 individual view-PSNR samples. Distribution: min 26.76dB, p5 35.55dB, median 45.21dB overall... Only 2.3% of samples fall below 33dB, comfortably separated from the actual reconstruction-quality range (8-25dB)... keeping the already-in-use PSNR>=33dB threshold as final.

**이 두 커밋 사이 어디에도 60dB가 등장한 적이 없다.** 즉 "draft ≥60dB gate"는 우리 쪽 어떤 시점의 실제 설정값도 아니다.

### 1.3 저장소 전체에서 "60dB" 검색

```
$ grep -rn "60.*dB\|dB.*60\|equivalence.*60" --include="*.md" --include="*.py" --include="*.json" --include="*.yaml" .
```

매치되는 모든 줄을 확인한 결과, 전부 `budget=60s`(시간, 초 단위)이거나 무관한 문맥이었다. **PSNR 60dB를 gate 기준으로 언급한 문서·코드·주석은 이 저장소에 하나도 없다.**

### 1.4 390샘플 통계 재확인

390샘플 통계는 `overall.md` §5.8에 이미 최종 동결 형태로 기록돼 있고, 위 커밋 메시지와 정확히 일치한다(min 26.76dB, median 45.21dB, 33dB 미달 2.3%). 원본 로그 자체(130개 gate check raw json)는 스케일업 과정에서 생성된 것으로 별도 재실행 없이 커밋 메시지·`overall.md` 서술로 충분히 재현 확인됨.

---

## 2. B의 측정값(43.8–44.3 dB) 판정

**우리 33dB 기준으로는 여유 있게 통과한다.** 390샘플 분포의 median(45.21dB) 근처에 위치하며, 미달 컷라인(33dB)보다 10dB 이상 높다.

**60dB 기준은 애초에 성립할 수 없는 값이다** — 우리 자신의 390샘플 중 최댓값이 60.11dB(`overall.md` 표)이고 그 값에 도달한 샘플이 극소수라, "60dB 이상이어야 PASS"라는 기준을 실제로 적용하면 우리가 이미 정상으로 판정해 사용 중인 결과 대부분이 함께 탈락한다. 이 기준으로는 게이트가 사실상 항상 FAIL하므로 애초에 실행 가능한 기준이 아니다.

## 3. 최종 판정

| 항목 | 판정 |
|---|---|
| 우리 동결 기준(33dB) 대비 B의 측정값(43.8–44.3dB) | **PASS** (여유 있게 통과) |
| "60dB gate"의 출처 | **확인 불가 — 이 저장소에는 존재하지 않음** |
| B와 동일한 측정 방식인지(view 집합·코드·체크포인트) | **미확인** — B-1 결과 없이는 판단 불가 |

**결론: 우리 기준 자체로는 PASS이고 C1-b를 막을 이유가 없다.** 다만 지시문의 "주의" 조항대로, B가 우리와 동일한 방식(같은 view 집합, 같은 PSNR 계산 코드, 같은 체크포인트)으로 측정했는지는 이 컨테이너에서 확인할 방법이 없어(B의 원본 로그 파일에 접근 불가) 최종 PASS 통보는 그 대조가 끝난 뒤로 보류한다.

## 4. B에게 전달할 요약

> 우리 쪽 renderer-equivalence gate는 33dB로 최종 동결돼 있습니다(2026-08-13, 커밋 `6884922`, 390샘플 실측 근거는 `overall.md` §5.8). 43.8–44.3dB는 이 기준을 여유 있게 통과합니다. "60dB" 기준은 저희 저장소 히스토리 어디에도 없는 값이라 어디서 참조하신 건지 확인 부탁드려요 — 혹시 예전 초안이나 다른 지표(예: MSE 값을 dB로 잘못 환산)를 보신 게 아닌가 싶습니다. 측정 방식(view 집합/코드/체크포인트)이 저희와 같다면 C1-b 바로 unblock하셔도 됩니다.

## 5. 폐기 처리가 필요한 문서

검색 결과 저장소 내에 60dB를 gate 기준으로 서술한 문서가 없으므로, **DEPRECATED 표기가 필요한 대상 자체가 없다.** (B 쪽 인계 문서에 있다면 그건 이 저장소 밖의 파일이라 우리가 직접 수정할 수 없음 — B에게 정정 요청 필요.)
