#!/usr/bin/env python3
"""Feed-forward(MVSplat/DepthSplat) Gaussian 출력을 gsplat 파라미터화로 변환한다 (C1-b 1단계).

MVSplat의 `Gaussians` 표현(확인 근거: `/data/Re-feem/code/mvsplat/src/model/{types,
encoder/common/gaussian_adapter}.py`, 2026-08-12 직접 코드 리딩):
- means: (N,3), covariances: (N,3,3) — 둘 다 **월드 좌표계**(camera-to-world 회전으로 이미
  변환됨, gaussian_adapter.forward()의 "Create world-space covariance matrices" 참고).
  즉 좌표계 변환은 필요 없다.
- harmonics: (N,3,d_sh), d_sh=(sh_degree+1)^2. RE10K/DTU 기본 config는 sh_degree=4 -> d_sh=25.
- opacities: (N,), 이미 [0,1] 확률(= "map_pdf_to_opacity" 출력) — sigmoid를 다시 씌우면
  안 되고, 우리 러너의 raw-logit 파라미터화에 넣으려면 inverse_sigmoid로 되돌려야 한다.

vanilla_3dgs_runner.py의 gsplat 파라미터화는 반대로:
- scales: log-space (rasterization에서 torch.exp)
- opacities: logit-space (rasterization에서 torch.sigmoid)
- quats: 정규화 전 raw quaternion (rasterization 안에서 norm으로 나눔)
- colors = cat([sh0, shN], dim=1), sh0=(N,1,3), shN=(N,d_sh-1,3)

이 파일은 그 변환만 담당한다. 데이터셋(DTU/RE10K)이나 렌더 등가성 검사는 다루지 않는다
(각각 `colmap_init.py`류 로더와 `renderer_equivalence.py`가 담당).
"""

from __future__ import annotations

import torch

EPS = 1e-8


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x / (1.0 - x))


def _matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) proper rotation matrix -> (..., 4) quaternion (w, x, y, z).

    표준 trace 기반 branchless 변환(4가지 경우 중 대각합이 가장 큰 항으로 분기하는 방식을
    where로 벡터화한 것 — PyTorch3D `matrix_to_quaternion`과 동일 계열의 공개된 수치안정
    공식). eigh가 만든 회전행렬은 이론상 orthogonal이지만 det=-1(반사)일 수 있어, 호출부
    (`covariance_to_scale_quat`)에서 미리 det 부호를 +1로 맞춘 뒤 이 함수에 넘긴다.
    """

    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]

    trace = m00 + m11 + m22

    def _case0():
        s = torch.sqrt((trace + 1.0).clamp_min(EPS)) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
        return torch.stack([w, x, y, z], dim=-1)

    def _case1():
        s = torch.sqrt((1.0 + m00 - m11 - m22).clamp_min(EPS)) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
        return torch.stack([w, x, y, z], dim=-1)

    def _case2():
        s = torch.sqrt((1.0 + m11 - m00 - m22).clamp_min(EPS)) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
        return torch.stack([w, x, y, z], dim=-1)

    def _case3():
        s = torch.sqrt((1.0 + m22 - m00 - m11).clamp_min(EPS)) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
        return torch.stack([w, x, y, z], dim=-1)

    cand0, cand1, cand2, cand3 = _case0(), _case1(), _case2(), _case3()

    use1 = (m00 > m11) & (m00 > m22)
    use2 = (~use1) & (m11 > m22)
    use3 = (~use1) & (~use2)
    trace_pos = trace > 0

    result = torch.where(trace_pos[..., None], cand0, torch.zeros_like(cand0))
    non_trace = ~trace_pos
    result = torch.where((non_trace & use1)[..., None], cand1, result)
    result = torch.where((non_trace & use2)[..., None], cand2, result)
    result = torch.where((non_trace & use3)[..., None], cand3, result)

    return result / result.norm(dim=-1, keepdim=True).clamp_min(EPS)


def covariance_to_scale_quat(covariances: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(N,3,3) 대칭 PSD 행렬을 (scale(N,3) 원래 크기, quat(N,4) w,x,y,z)로 분해한다.

    covariance = R @ diag(scale^2) @ R^T 이므로 eigh(고유값 오름차순, 고유벡터 정규직교)로
    scale = sqrt(eigval), R = eigvecs. eigh 결과는 orthogonal이지만 det=-1(반사)일 수 있어
    마지막 열 부호를 뒤집어 proper rotation(det=+1)으로 강제한다.
    """

    eigvals, eigvecs = torch.linalg.eigh(covariances.double())
    eigvals = eigvals.clamp_min(0.0)
    scale = torch.sqrt(eigvals)

    det = torch.linalg.det(eigvecs)
    flip = (det < 0).unsqueeze(-1)  # (...,1), last-column과 broadcast하기 위해 차원 유지
    # 마지막 열만 부호 반전하면 나머지 두 열은 그대로 orthogonal 유지된 채 det만 +1로 바뀐다.
    eigvecs_fixed = eigvecs.clone()
    last_col = eigvecs_fixed[..., :, 2]
    eigvecs_fixed[..., :, 2] = torch.where(flip, -last_col, last_col)

    quat = _matrix_to_quaternion(eigvecs_fixed.float())
    return scale.float(), quat


def gaussians_to_gsplat_params(
    means: torch.Tensor,
    covariances: torch.Tensor,
    harmonics: torch.Tensor,
    opacities: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.nn.Parameter]:
    """MVSplat류 `Gaussians`(월드 좌표) -> vanilla_3dgs_runner.py 파라미터화.

    입력 shape (배치 차원 없이 flatten된 상태로 받는다 — 호출부에서 `rearrange(... "b n ... -> (b n) ...")`):
    - means: (N,3), covariances: (N,3,3), harmonics: (N,3,d_sh), opacities: (N,)
    """

    means = means.to(device)
    covariances = covariances.to(device)
    harmonics = harmonics.to(device)
    opacities = opacities.to(device)

    scale, quat = covariance_to_scale_quat(covariances)
    log_scale = torch.log(scale.clamp_min(1e-8))

    # harmonics: (N,3,d_sh) -> 우리 러너 관례 (N,d_sh,3). 0번째 밴드(DC)가 sh0, 나머지가 shN.
    harmonics_reordered = harmonics.permute(0, 2, 1).contiguous()  # (N, d_sh, 3)
    sh0 = harmonics_reordered[:, :1, :]
    shN = harmonics_reordered[:, 1:, :]

    opacities_logit = inverse_sigmoid(opacities)

    return {
        "means": torch.nn.Parameter(means.float()),
        "scales": torch.nn.Parameter(log_scale.float()),
        "quats": torch.nn.Parameter(quat.float()),
        "opacities": torch.nn.Parameter(opacities_logit.float()),
        "sh0": torch.nn.Parameter(sh0.float()),
        "shN": torch.nn.Parameter(shN.float()),
    }
