"""Registry riêng cho Module D — Asymmetric Fusion Rule (Base ≠ Detail).

Mỗi entry: (base_rule_factory, detail_rule_factory, pixel_select).
- base_rule_factory:   () -> nn.Module thay phép `+` ở Base path
- detail_rule_factory: () -> nn.Module thay phép `+` ở Detail path
- pixel_select:        'max' (paper) hoặc 'saliency' (B.3 winner) — chọn target cho L_int

Training protocol (xem PROGRESS.md Module D):
- Load pretrained E/D từ `CDDFuse_MIF.pth`
- FREEZE Encoder + Decoder (requires_grad=False)
- Chỉ train Fusion modules (BaseFuseLayer + DetailFuseLayer + base_rule + detail_rule)
- Phase II only, ~30 epoch trên 738 cặp Harvard

Stage 1 (D-Base sweep): Detail = IdentitySum (paper), sweep Base rule
Stage 2 (D-Detail sweep): Base = IdentitySum (paper), sweep Detail rule
Stage 3 (Combined): best Base × best Detail
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch.nn as nn

from .base_rules import (L1NormBaseFuse, LocalEnergyBaseFuse,
                          LocalEntropyBaseFuse, VisualSaliencyMapFuse,
                          WeightedAvgScalarFuse)
from .detail_rules import (L1NormDetailFuse, LocalEnergyDetailFuse,
                            SaliencyDetailGate, SMLFuse, SoftMaxAbsFuse,
                            SoftPCNNFuse, SpatialFrequencyFuse)
from .modules import GatedFuseLayer, IdentitySum


# Type alias
ModuleFactory = Callable[[], nn.Module]
AsymVariant = Tuple[ModuleFactory, ModuleFactory, str]


# Pixel select default: dùng 'saliency' vì đã CONFIRM ở Module B (B.3 winner)
# có thể override khi cần ablation pixel rule
PIXEL_SELECT = "saliency"


VARIANT_REGISTRY_ASYM: Dict[str, AsymVariant] = {
    # ==================== Stage 1: D-Base sweep (Detail = Sum) ====================
    "AsymD-DB0-BaseSum":         (lambda: IdentitySum(),               lambda: IdentitySum(), PIXEL_SELECT),
    "AsymD-DB1-BaseGated":       (lambda: GatedFuseLayer(64, scale=2.0), lambda: IdentitySum(), PIXEL_SELECT),
    "AsymD-DB2-BaseL1Norm":      (lambda: L1NormBaseFuse(),            lambda: IdentitySum(), PIXEL_SELECT),
    "AsymD-DB3-BaseVSM":         (lambda: VisualSaliencyMapFuse(64),   lambda: IdentitySum(), PIXEL_SELECT),
    "AsymD-DB4-BaseLocalEnergy": (lambda: LocalEnergyBaseFuse(),       lambda: IdentitySum(), PIXEL_SELECT),
    "AsymD-DB5-BaseLocalEntropy":(lambda: LocalEntropyBaseFuse(),      lambda: IdentitySum(), PIXEL_SELECT),
    "AsymD-DB6-BaseWeightedAvg": (lambda: WeightedAvgScalarFuse(),     lambda: IdentitySum(), PIXEL_SELECT),

    # ==================== Stage 2: D-Detail sweep (Base = Sum) ====================
    "AsymD-DD0-DetailSum":         (lambda: IdentitySum(), lambda: IdentitySum(),                 PIXEL_SELECT),  # = DB0
    "AsymD-DD1-DetailGated":       (lambda: IdentitySum(), lambda: GatedFuseLayer(64, scale=2.0), PIXEL_SELECT),
    "AsymD-DD2-DetailMaxAbs":      (lambda: IdentitySum(), lambda: SoftMaxAbsFuse(64, 0.1),       PIXEL_SELECT),
    "AsymD-DD3-DetailSaliency":    (lambda: IdentitySum(), lambda: SaliencyDetailGate(64),        PIXEL_SELECT),
    "AsymD-DD4-DetailSF":          (lambda: IdentitySum(), lambda: SpatialFrequencyFuse(64),      PIXEL_SELECT),
    "AsymD-DD5-DetailLocalEnergy": (lambda: IdentitySum(), lambda: LocalEnergyDetailFuse(),       PIXEL_SELECT),
    "AsymD-DD6-DetailSML":         (lambda: IdentitySum(), lambda: SMLFuse(64),                   PIXEL_SELECT),
    "AsymD-DD7-DetailL1Norm":      (lambda: IdentitySum(), lambda: L1NormDetailFuse(),            PIXEL_SELECT),
    "AsymD-DD8-DetailPCNNsoft":    (lambda: IdentitySum(), lambda: SoftPCNNFuse(64, T=10),        PIXEL_SELECT),

    # ==================== Stage 3: Combined (best × best) — placeholder ====================
    # Đặt thêm khi xác định winner Stage 1 + Stage 2
    # Ví dụ:
    # "AsymD-Combined-VSM-Saliency": (lambda: VisualSaliencyMapFuse(64), lambda: SaliencyDetailGate(64), PIXEL_SELECT),
}


def build_asym_variant(name: str) -> AsymVariant:
    """Build (base_module, detail_module, pixel_select) từ name.

    Auto-strip prefix 'CDDFuse-' nếu có (theo convention các registry khác).
    """
    if name.startswith("CDDFuse-"):
        name = name[len("CDDFuse-"):]
    if name not in VARIANT_REGISTRY_ASYM:
        raise KeyError(
            f"Unknown asymmetric variant '{name}'. "
            f"Available: {list(VARIANT_REGISTRY_ASYM)}"
        )
    base_fac, detail_fac, pixel_select = VARIANT_REGISTRY_ASYM[name]
    return base_fac(), detail_fac(), pixel_select


def list_asym_variants() -> list:
    return list(VARIANT_REGISTRY_ASYM)
