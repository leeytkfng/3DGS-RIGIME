# 데이터셋 인용·라이선스 — 논문용 최종 정리 (2026-08-13)

체크리스트 "RE10K citation/license 문구 논문용 정리" 항목. 논문 Data/Acknowledgments 절에
그대로 옮겨 쓸 수 있게 정리했다. 출처는 웹서치로 실측 확인(각 절 끝에 링크 표기) — 데이터
자체 획득 경위는 `/data/Re-feem/datasets/{re10k,dl3dv}/SOURCE.md`에 이미 기록돼 있고, 이
문서는 그중 "논문에 어떻게 쓸지"만 최종 정리한 것이다.

---

## RE10K (RealEstate10K)

**원 논문 인용**:

> T. Zhou, R. Tucker, J. Flynn, G. Fyffe, and N. Snavely. "Stereo Magnification: Learning View
> Synthesis using Multiplane Images." *ACM Transactions on Graphics (Proc. SIGGRAPH)*, 2018.
> arXiv:1805.09817.

BibTeX:

```bibtex
@article{zhou2018stereo,
  title={Stereo magnification: Learning view synthesis using multiplane images},
  author={Zhou, Tinghui and Tucker, Richard and Flynn, John and Fyffe, Graham and Snavely, Noah},
  journal={ACM Transactions on Graphics (Proc. SIGGRAPH)},
  year={2018}
}
```

**라이선스**: Google LLC, **CC BY 4.0** (Creative Commons Attribution 4.0 International) —
공식 출처 명시(google.github.io/realestate10k) 확인. 재배포·변형·상업적 이용 모두 허용, 저작자
표시(위 인용)만 요구.

**우리가 실제로 쓴 경로에 대한 주의**: 114 scene 중 41개는 pixelSplat 공식 small subset
(Google Drive), 73개는 Hugging Face `Hualingchu/RealEstate10K_test`(개인 재업로드 mirror,
원 저작자가 명시한 공식 배포처는 아님, `.torch` chunk 포맷은 pixelSplat/MVSplat과 동일)에서
받았다. 원 데이터셋의 라이선스(CC BY 4.0)는 그대로 적용되지만, **mirror 자체의 신뢰성은
원저작자 보증이 아니라는 점을 방법론(§Data) 절에 한 문장으로 명시할 것** — "test chunk 포맷
검증(MVSplat 공식 checkpoint로 mean PSNR 22.4dB, 기존 probe 25.6dB와 동일 정상 범위)을
거쳤다"는 이미 확보한 근거(SOURCE.md)를 함께 적으면 충분하다.

Sources: [RealEstate10K](https://google.github.io/realestate10k/), [Download/License page](https://google.github.io/realestate10k/download.html)

---

## DL3DV-10K

**원 논문 인용**:

> L. Ling, Y. Sheng, Z. Tu, W. Zhao, C. Xin, K. Wan, L. Yu, Q. Guo, Z. Yu, Y. Lu, et al.
> "DL3DV-10K: A Large-Scale Scene Dataset for Deep Learning-based 3D Vision." *Proceedings of
> the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2024, pp. 22160–22169.
> arXiv:2312.16256.

BibTeX:

```bibtex
@inproceedings{ling2024dl3dv,
  title={DL3DV-10K: A Large-Scale Scene Dataset for Deep Learning-based 3D Vision},
  author={Ling, Lu and Sheng, Yichen and Tu, Zhi and Zhao, Wentian and Xin, Cheng and Wan, Kun
    and Yu, Lantao and Guo, Qianyu and Yu, Zixun and Lu, Yawen and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={22160--22169},
  year={2024}
}
```

**라이선스**: **CC BY-NC 4.0**(비상업적 이용만 허용) + 자체 Terms of Use 별도 존재
(`DL3DV-10K/Dataset` GitHub repo에 명시) — **학술 연구 목적은 문제없지만, NC 조항이 있다는
걸 논문에 한 줄 명시할 것.** 접근은 Hugging Face gated repo(`DL3DV/DL3DV-ALL-480P`)를 통해
계정 인증 후 이용약관 동의로 받았다(2026-08-10, SOURCE.md 기록).

**우리가 실제로 쓴 경로에 대한 주의**: pilot 25 scene은 `DL3DV-ALL-480P`(전체 10,581 scene
중 자체 랜덤 샘플링)에서 왔고, 공식 평가 split(`DL3DV-Benchmark`, 141 scene)과는 중복 0으로
이미 확인됐다(SOURCE.md) — leakage 없음. 다만 이 25개 자체가 "표준 benchmark"는 아니라는 점을
방법론 절에 명시해야 한다(우리 자체 pilot subset이라는 사실).

Sources: [arXiv:2312.16256](https://arxiv.org/abs/2312.16256), [CVPR 2024 Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Ling_DL3DV-10K_A_Large-Scale_Scene_Dataset_for_Deep_Learning-based_3D_Vision_CVPR_2024_paper.html), [DL3DV-10K/Dataset](https://github.com/DL3DV-10K/Dataset)

---

## 논문에 넣을 한 단락 (초안)

> We use RealEstate10K [Zhou et al. 2018] (CC BY 4.0) as our primary benchmark and DL3DV-10K
> [Ling et al. 2024] (CC BY-NC 4.0, non-commercial) as a secondary dataset. For RealEstate10K,
> we use the official test split (114 scenes assembled from the pixelSplat-hosted subset and a
> community mirror in the same chunk format, verified against an official MVSplat checkpoint to
> confirm data integrity). For DL3DV-10K, we use a 25-scene pilot subset randomly sampled from
> `DL3DV-ALL-480P`, confirmed to have zero overlap with the official `DL3DV-Benchmark` evaluation
> split.

## 남은 확인 사항 (제출 전)

- [ ] DL3DV-10K GitHub의 정확한 "Terms of Use" 전문을 논문 제출 시점에 다시 확인(약관이 바뀔 수 있음)
- [ ] RE10K 114 scene 중 mirror 출처(73개)를 그대로 공개 데이터로 재배포할 계획이 있다면(예: 재현성 패키지) CC BY 4.0 표시 의무 이행 방법 확인
