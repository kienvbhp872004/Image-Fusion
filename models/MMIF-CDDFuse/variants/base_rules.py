"""Base fusion rules cho Module D Stage 1 — D-Base ablation.

Mỗi class thay phép `+` trong:
    f_F_B = base_fuse(rule(f_V_B, f_I_B))

Convention: forward(a, b) -> tensor cùng shape a.
Init phải đảm bảo epoch-0 ≈ baseline (sum/average) để load pretrained E/D về có hành vi ổn định.

Các rule này tập trung vào BASE component (low-freq, structural). Mục tiêu: bảo toàn
cấu trúc giải phẫu chung 2 modality, tránh artifact biên.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "L1NormBaseFuse",
    "VisualSaliencyMapFuse",
    "LocalEnergyBaseFuse",
    "LocalEntropyBaseFuse",
    "WeightedAvgScalarFuse",
]


# ---------- DB.2: L1-norm based weighted average ----------
class L1NormBaseFuse(nn.Module):
    """Weighted average theo L1-norm của feature từng pixel.

        w_a = ||a||_1 / (||a||_1 + ||b||_1 + eps)   [B, 1, H, W]
        out = w_a * a + (1 - w_a) * b

    Tham số: 0 (rule-based, không học).
    Trực giác: modality nào có feature magnitude lớn hơn → trọng số cao hơn.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        norm_a = a.abs().sum(dim=1, keepdim=True)
        norm_b = b.abs().sum(dim=1, keepdim=True)
        w_a = norm_a / (norm_a + norm_b + self.eps)
        return w_a * a + (1.0 - w_a) * b


# ---------- DB.3: Visual Saliency Map weighted ----------
class VisualSaliencyMapFuse(nn.Module):
    """VSM (Visual Saliency Map) — gán trọng số cao hơn cho vùng tương phản cao.

    Approximation từ Wang & Liu 2008 (cho ảnh xám):
        VSM(p) = sum_{q in window} |I(p) - I(q)| * weight(I(q))
    Ở đây áp dụng trên feature map (đa channel) với approximation:
        VSM(a) = local_std(a) [B, 1, H, W]

    Trọng số: w_a = VSM(a) / (VSM(a) + VSM(b) + eps).

    Params: 1 Conv 3x3 -> 1ch để học saliency map từ feature (4K params).
    """

    def __init__(self, dim: int = 64, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.saliency_conv = nn.Conv2d(dim, 1, kernel_size=kernel, padding=pad, bias=True)
        # Init: weight nhỏ ngẫu nhiên + bias = 0 → epoch-0 VSM ≈ 0 → w ≈ 0.5 (average baseline)
        nn.init.normal_(self.saliency_conv.weight, std=0.01)
        nn.init.zeros_(self.saliency_conv.bias)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Local saliency (tương phản với neighbor) — dùng conv để học
        s_a = self.saliency_conv(a).abs()  # [B, 1, H, W]
        s_b = self.saliency_conv(b).abs()
        w_a = s_a / (s_a + s_b + 1e-8)
        return w_a * a + (1.0 - w_a) * b


# ---------- DB.4: Local Energy weighted ----------
class LocalEnergyBaseFuse(nn.Module):
    """Local Energy = sum_{neighbor} feature² trong cửa sổ W×W.

        E(a)[p] = avg_pool(a², window=W)   [B, C, H, W]
        w_a = sum_C E(a) / (sum_C E(a) + sum_C E(b) + eps)   [B, 1, H, W]
        out = w_a * a + (1 - w_a) * b

    Params: 0 (purely rule-based).
    Trực giác: vùng có energy cao hơn = vùng có thông tin → ưu tiên.
    """

    def __init__(self, window: int = 5):
        super().__init__()
        self.window = window
        self.pad = window // 2

    def _local_energy(self, x: torch.Tensor) -> torch.Tensor:
        sq = x * x
        # avg_pool2d giữ shape khi padding đúng
        e = F.avg_pool2d(sq, kernel_size=self.window, stride=1, padding=self.pad)
        return e.sum(dim=1, keepdim=True)  # [B, 1, H, W]

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        e_a = self._local_energy(a)
        e_b = self._local_energy(b)
        w_a = e_a / (e_a + e_b + 1e-8)
        return w_a * a + (1.0 - w_a) * b


# ---------- DB.5: Local Entropy weighted ----------
class LocalEntropyBaseFuse(nn.Module):
    """Local Entropy approximation — vùng nào "đa dạng" thông tin hơn.

    Approx entropy bằng:
        H(a) ≈ -sum p log p  với p = softmax(local feature) trên kênh
        Hoặc: variance của feature trong cửa sổ (proxy của entropy)

    Dùng local variance làm proxy (tính toán đơn giản, monotone với H trên Gaussian).
        Var(a)[p] = avg(a²)[p] - (avg(a)[p])²
        H_a = sum_C Var(a)   [B, 1, H, W]
        w_a = H_a / (H_a + H_b + eps)

    Params: 0.
    """

    def __init__(self, window: int = 5):
        super().__init__()
        self.window = window
        self.pad = window // 2

    def _local_var(self, x: torch.Tensor) -> torch.Tensor:
        mean = F.avg_pool2d(x, self.window, stride=1, padding=self.pad)
        mean_sq = F.avg_pool2d(x * x, self.window, stride=1, padding=self.pad)
        var = (mean_sq - mean * mean).clamp(min=0.0)
        return var.sum(dim=1, keepdim=True)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        v_a = self._local_var(a)
        v_b = self._local_var(b)
        w_a = v_a / (v_a + v_b + 1e-8)
        return w_a * a + (1.0 - w_a) * b


# ---------- DB.6: Learnable scalar weighted average ----------
class WeightedAvgScalarFuse(nn.Module):
    """Weighted average với 1 scalar học được.

        out = α * a + (1 - α) * b,   α = sigmoid(logit) ∈ (0, 1)

    Params: 1 scalar.
    Init: logit = 0 → α = 0.5 → epoch-0 = average baseline.
    """

    def __init__(self):
        super().__init__()
        self.logit = nn.Parameter(torch.zeros(1))

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        alpha = torch.sigmoid(self.logit)
        return alpha * a + (1.0 - alpha) * b
