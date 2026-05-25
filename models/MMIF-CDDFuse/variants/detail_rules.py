"""Detail fusion rules cho Module D Stage 2 — D-Detail ablation.

Mỗi class thay phép `+` trong:
    f_F_D = detail_fuse(rule(f_V_D, f_I_D))

Convention: forward(a, b) -> tensor cùng shape a.
Init phải đảm bảo epoch-0 ≈ baseline để load pretrained E/D ổn định.

Các rule này tập trung vào DETAIL component (high-freq, texture). Mục tiêu: chọn
tín hiệu mạnh nhất từ mỗi modality (texture mô mềm MRI, rim xương CT, ...).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "SoftMaxAbsFuse",
    "SaliencyDetailGate",
    "SpatialFrequencyFuse",
    "LocalEnergyDetailFuse",
    "SMLFuse",
    "L1NormDetailFuse",
    "SoftPCNNFuse",
]


# ---------- DD.2: Soft choose-max-abs ----------
class SoftMaxAbsFuse(nn.Module):
    """Soft version của max-abs rule (truyền thống cho high-freq):

        w_a = softmax(|a|/τ) component-wise vs |b|/τ
            = sigmoid((|a| - |b|) / τ)        [B, C, H, W]
        out = w_a * a + (1 - w_a) * b

    Khi τ → 0: tiến tới hard choose-max (a nếu |a| > |b|).
    Khi τ → ∞: tiến tới average.

    Init: 1 Conv 1×1 từ |a|-|b| → temperature/score (8K params dim=64).
    epoch-0: weight Conv = 0, bias = 0 → score = 0 → w = 0.5 → average.
    """

    def __init__(self, dim: int = 64, temperature: float = 0.1):
        super().__init__()
        self.tau = temperature
        # Conv 1×1 học refine từ |a| - |b| → score
        self.score = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        diff = a.abs() - b.abs()
        score = self.score(diff)  # [B, C, H, W]
        w_a = torch.sigmoid(score / self.tau)
        return w_a * a + (1.0 - w_a) * b


# ---------- DD.3: Saliency-guided detail gate ----------
class SaliencyDetailGate(nn.Module):
    """Gating dựa trên gradient của feature (đồng bộ với saliency-guided pixel loss).

        grad_a = sobel-like gradient of a   [B, C, H, W]
        grad_b = sobel-like gradient of b
        score = Conv(|grad_a| - |grad_b|)
        w_a = sigmoid(score)
        out = w_a * a + (1 - w_a) * b

    Params: Sobel filter cố định + Conv1x1 refine (10K).
    Init Conv = 0 → epoch-0 = average.
    """

    def __init__(self, dim: int = 64):
        super().__init__()
        # Sobel filter cho gradient (cố định, không train)
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        # Replicate cho mỗi channel → depthwise conv
        self.register_buffer('sobel_x', sobel_x.repeat(dim, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.repeat(dim, 1, 1, 1))
        self.dim = dim
        # Conv 1x1 refine
        self.refine = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        nn.init.zeros_(self.refine.weight)
        nn.init.zeros_(self.refine.bias)

    def _grad_mag(self, x: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(x, self.sobel_x, padding=1, groups=self.dim)
        gy = F.conv2d(x, self.sobel_y, padding=1, groups=self.dim)
        return torch.sqrt(gx * gx + gy * gy + 1e-8)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        g_a = self._grad_mag(a)
        g_b = self._grad_mag(b)
        score = self.refine(g_a - g_b)
        w_a = torch.sigmoid(score)
        return w_a * a + (1.0 - w_a) * b


# ---------- DD.4: Spatial Frequency weighted ----------
class SpatialFrequencyFuse(nn.Module):
    """Spatial Frequency = sqrt(RF² + CF²) đo độ "động" của texture.

        RF(p) = a(p) - a(p-1)  (row gradient)
        CF(p) = a(p) - a(p-W)  (column gradient)
        SF(a) = avg_pool(sqrt(RF² + CF²), window)   [B, C, H, W]
        w_a = SF(a) / (SF(a) + SF(b) + eps)
        out = w_a * a + (1 - w_a) * b

    Params: 0 (rule-based) + 1 Conv 1x1 refine (4K).
    """

    def __init__(self, dim: int = 64, window: int = 5):
        super().__init__()
        self.window = window
        self.pad = window // 2
        self.refine = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        nn.init.zeros_(self.refine.weight)
        nn.init.zeros_(self.refine.bias)

    def _sf(self, x: torch.Tensor) -> torch.Tensor:
        # Row gradient
        rf = torch.zeros_like(x)
        rf[..., :, 1:] = x[..., :, 1:] - x[..., :, :-1]
        # Column gradient
        cf = torch.zeros_like(x)
        cf[..., 1:, :] = x[..., 1:, :] - x[..., :-1, :]
        sf = torch.sqrt(rf * rf + cf * cf + 1e-8)
        sf = F.avg_pool2d(sf, self.window, stride=1, padding=self.pad)
        return sf

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        sf_a = self._sf(a)
        sf_b = self._sf(b)
        score = self.refine(sf_a - sf_b)
        w_a = torch.sigmoid(score)
        return w_a * a + (1.0 - w_a) * b


# ---------- DD.5: Local Energy weighted (giống Base nhưng tính per-channel) ----------
class LocalEnergyDetailFuse(nn.Module):
    """Local Energy per-channel cho Detail.

        E(a)[c, p] = avg_pool(a[c]², window)   [B, C, H, W]
        w_a = E(a) / (E(a) + E(b) + eps)       [B, C, H, W] (per channel!)
        out = w_a * a + (1 - w_a) * b

    Khác Base: weight per-channel (Detail mỗi channel ~ 1 frequency band).
    Params: 0.
    """

    def __init__(self, window: int = 5):
        super().__init__()
        self.window = window
        self.pad = window // 2

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        e_a = F.avg_pool2d(a * a, self.window, stride=1, padding=self.pad)
        e_b = F.avg_pool2d(b * b, self.window, stride=1, padding=self.pad)
        w_a = e_a / (e_a + e_b + 1e-8)
        return w_a * a + (1.0 - w_a) * b


# ---------- DD.6: Sum-Modified-Laplacian ----------
class SMLFuse(nn.Module):
    """Sum-Modified-Laplacian (SML) — đo độ nét / sharpness.

        ML(p) = |2a(p) - a(p-1) - a(p+1)| + |2a(p) - a(p-W) - a(p+W)|
        SML(a) = sum_{p in window} ML(p)
        w_a = SML(a) / (SML(a) + SML(b) + eps)

    Phổ biến cho multi-focus fusion.
    Params: 1 Conv 1x1 refine (4K).
    """

    def __init__(self, dim: int = 64, window: int = 5):
        super().__init__()
        self.window = window
        self.pad = window // 2
        self.refine = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        nn.init.zeros_(self.refine.weight)
        nn.init.zeros_(self.refine.bias)

    def _ml(self, x: torch.Tensor) -> torch.Tensor:
        # 2x - x_left - x_right (horizontal)
        ml_h = torch.zeros_like(x)
        ml_h[..., :, 1:-1] = (2 * x[..., :, 1:-1] - x[..., :, :-2] - x[..., :, 2:]).abs()
        # 2x - x_up - x_down (vertical)
        ml_v = torch.zeros_like(x)
        ml_v[..., 1:-1, :] = (2 * x[..., 1:-1, :] - x[..., :-2, :] - x[..., 2:, :]).abs()
        ml = ml_h + ml_v
        # Sum trong window
        sml = F.avg_pool2d(ml, self.window, stride=1, padding=self.pad)
        return sml

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        s_a = self._ml(a)
        s_b = self._ml(b)
        score = self.refine(s_a - s_b)
        w_a = torch.sigmoid(score)
        return w_a * a + (1.0 - w_a) * b


# ---------- DD.7: L1-norm weighted (Detail version, per-channel) ----------
class L1NormDetailFuse(nn.Module):
    """L1-norm per-channel cho Detail (khác Base ở chỗ không sum_C):

        w_a[c] = |a[c]| / (|a[c]| + |b[c]| + eps)    [B, C, H, W]
        out = w_a * a + (1 - w_a) * b

    Params: 0.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        w_a = a.abs() / (a.abs() + b.abs() + self.eps)
        return w_a * a + (1.0 - w_a) * b


# ---------- DD.8: Soft PCNN ----------
class SoftPCNNFuse(nn.Module):
    """Soft PCNN (differentiable) — đếm fire frequency qua T iteration.

    Simplified PCNN với σ thay step function:
        F(t) = x (feeding input = feature)
        L(t) = depthwise_conv(Y(t-1)) (linking from neighbors)
        U(t) = F * (1 + β*L)
        θ(t) = θ(t-1) * exp(-α_θ) + V_θ * Y(t-1)
        Y(t) = sigmoid((U - θ) / temperature)   # soft step
    Sau T iter: fire_freq(a) = Σ_t Y(t) ∈ [0, T]

        w_a = fire_freq(a) / (fire_freq(a) + fire_freq(b) + eps)
        out = w_a * a + (1 - w_a) * b

    Params: 1 linking conv 3x3 depthwise (1.1K) + 3 scalar (β, α_θ, V_θ) = 12K total.
    """

    def __init__(self, dim: int = 64, T: int = 10, temperature: float = 0.5,
                 init_beta: float = 0.2, init_alpha: float = 0.5, init_vth: float = 1.0):
        super().__init__()
        self.dim = dim
        self.T = T
        self.tau = temperature
        # Linking weights — depthwise conv 3x3
        self.linking = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        nn.init.constant_(self.linking.weight, 1.0 / 9.0)  # avg pool init
        # PCNN parameters (học)
        self.beta = nn.Parameter(torch.tensor(init_beta))
        self.alpha_theta = nn.Parameter(torch.tensor(init_alpha))
        self.v_theta = nn.Parameter(torch.tensor(init_vth))

    def _fire_freq(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize x về [0, 1] approximation
        F_in = x.abs()
        Y = torch.zeros_like(F_in)
        theta = torch.zeros_like(F_in)
        fire_sum = torch.zeros_like(F_in)
        for _ in range(self.T):
            L = self.linking(Y)
            U = F_in * (1.0 + self.beta * L)
            theta = theta * torch.exp(-self.alpha_theta) + self.v_theta * Y
            Y = torch.sigmoid((U - theta) / self.tau)
            fire_sum = fire_sum + Y
        return fire_sum

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        fa = self._fire_freq(a)
        fb = self._fire_freq(b)
        w_a = fa / (fa + fb + 1e-8)
        return w_a * a + (1.0 - w_a) * b
