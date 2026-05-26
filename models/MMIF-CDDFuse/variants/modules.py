"""
Variant modules for CDDFuse fusion-rule ablation (Module A).

Mỗi class ở đây thay thế phép `+` trong:
    f_f_b = base_fuse(f_v_b + f_i_b)
    f_f_d = detail_fuse(f_v_d + f_i_d)

Convention: forward(feat_a, feat_b) -> tensor cùng shape feat_a.
Init phải đảm bảo behavior epoch-0 ≈ baseline (sum hoặc avg) để light retrain hội tụ nhanh.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentitySum(nn.Module):
    """Baseline: behavior gốc của paper, fuse = A + B. Không có tham số."""

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        return feat_a + feat_b


class GatedFuseLayer(nn.Module):
    """
    Soft per-channel × per-spatial gate thay cho A + B.

        g     = sigmoid(Conv1x1([A, B]))    # shape [B, dim, H, W]
        out   = scale * (g * A + (1 - g) * B)

    Init: Conv weight và bias = 0  =>  g = sigmoid(0) = 0.5
          => epoch-0: out = scale * (A + B) / 2
          Với scale=1.0 (Module A default): out = 0.5*(A+B) [0.5× của paper sum]
          Với scale=2.0 (Module D — FROZEN E/D):  out = A+B [match paper sum]

    Module A (Combined-Gated-Saliency, AG-45ep) dùng scale=1.0 (E/D fine-tune được).
    Module D dùng scale=2.0 vì E/D frozen — phải giữ magnitude paper baseline.

    Tham số: dim*2*dim*1*1 + dim ≈ 8K cho dim=64.
    """

    def __init__(self, dim: int = 64, scale: float = 1.0):
        super().__init__()
        self.gate = nn.Conv2d(in_channels=dim * 2, out_channels=dim, kernel_size=1, bias=True)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.scale = scale

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(torch.cat([feat_a, feat_b], dim=1)))
        return self.scale * (g * feat_a + (1.0 - g) * feat_b)


class CrossAttnFuse(nn.Module):
    """
    Bidirectional channel-wise cross-attention (Restormer MDTA style).

    A queries B's channels and vice versa, results blended via 1×1 projection.
    Channel-attention (C×C, không phải HW×HW) -> linear in HW, scale tốt với ảnh y khoa.

    Init: proj_out weight và bias = 0  =>  out = 0.5·(A + B)  ≈  baseline behavior epoch-0.
    Tham số ~50K cho dim=64, num_heads=4.
    """

    def __init__(self, dim: int = 64, num_heads: int = 4, bias: bool = True):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must divide num_heads {num_heads}"
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv_a       = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_b       = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_a_dconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1,
                                     groups=dim * 3, bias=bias)
        self.qkv_b_dconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1,
                                     groups=dim * 3, bias=bias)
        self.proj_out    = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

        # Identity init: epoch-0 output = 0.5*(A+B), close to baseline sum (after rescale).
        nn.init.zeros_(self.proj_out.weight)
        if bias:
            nn.init.zeros_(self.proj_out.bias)

    def _to_heads(self, x: torch.Tensor, B: int, H: int, W: int) -> torch.Tensor:
        # (B, C, H, W) -> (B, num_heads, C/num_heads, H*W)
        C = x.shape[1]
        return x.reshape(B, self.num_heads, C // self.num_heads, H * W)

    def _from_heads(self, x: torch.Tensor, B: int, C: int, H: int, W: int) -> torch.Tensor:
        return x.reshape(B, C, H, W)

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat_a.shape
        qkv_a = self.qkv_a_dconv(self.qkv_a(feat_a))
        qkv_b = self.qkv_b_dconv(self.qkv_b(feat_b))
        qa, ka, va = qkv_a.chunk(3, dim=1)
        qb, kb, vb = qkv_b.chunk(3, dim=1)

        qa = self._to_heads(qa, B, H, W); ka = self._to_heads(ka, B, H, W); va = self._to_heads(va, B, H, W)
        qb = self._to_heads(qb, B, H, W); kb = self._to_heads(kb, B, H, W); vb = self._to_heads(vb, B, H, W)

        # L2-normalize on channel dim for stable channel-attention (Restormer convention)
        qa = F.normalize(qa, dim=-1); ka = F.normalize(ka, dim=-1)
        qb = F.normalize(qb, dim=-1); kb = F.normalize(kb, dim=-1)

        # Cross-attention: A queries B
        attn_ab = (qa @ kb.transpose(-2, -1)) * self.temperature
        attn_ab = attn_ab.softmax(dim=-1)
        out_ab  = self._from_heads(attn_ab @ vb, B, C, H, W)

        # Cross-attention: B queries A
        attn_ba = (qb @ ka.transpose(-2, -1)) * self.temperature
        attn_ba = attn_ba.softmax(dim=-1)
        out_ba  = self._from_heads(attn_ba @ va, B, C, H, W)

        out = self.proj_out(torch.cat([out_ab, out_ba], dim=1))
        return 0.5 * (feat_a + feat_b) + out


class ChannelMoEFuse(nn.Module):
    """
    Per-channel Mixture-of-Experts fusion. 3 experts per channel:
        expert_0 = A
        expert_1 = B
        expert_2 = (A + B) / 2
    Router: global avg-pool -> small MLP -> per-channel softmax over 3 experts.

    Lấy cảm hứng từ TC-MoA (Yang et al., CVPR 2024) — adapt routing pattern xuống
    fusion-stage thay vì task-adapter level.

    Init: last linear weight và bias = 0  =>  softmax(0,0,0) = (1/3, 1/3, 1/3)
          =>  output = (1/3)A + (1/3)B + (1/3)·0.5(A+B) = 0.5·(A + B) ≈ baseline.
    Params: ~5K cho dim=64, hidden=16 (per module).
    """

    def __init__(self, dim: int = 64, hidden: int = 16):
        super().__init__()
        self.dim = dim
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),         # B, 2C, 1, 1
            nn.Flatten(),                    # B, 2C
            nn.Linear(dim * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim * 3),      # 3 experts × C channels
        )
        # Identity init: uniform routing -> output = 0.5*(A + B)
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat_a.shape
        x = torch.cat([feat_a, feat_b], dim=1)               # B, 2C, H, W
        logits = self.router(x)                              # B, 3C
        weights = logits.reshape(B, 3, C).softmax(dim=1)     # B, 3, C
        w_a = weights[:, 0].reshape(B, C, 1, 1)
        w_b = weights[:, 1].reshape(B, C, 1, 1)
        w_m = weights[:, 2].reshape(B, C, 1, 1)
        mixed = 0.5 * (feat_a + feat_b)
        return w_a * feat_a + w_b * feat_b + w_m * mixed
