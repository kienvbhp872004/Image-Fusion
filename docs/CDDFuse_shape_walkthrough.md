# CDDFuse / CDDFuse-AG — Walkthrough chi tiết theo SHAPE

> Trace dữ liệu qua từng layer, với shape cụ thể `[B, C, H, W]` ở mỗi bước.
> Ví dụ minh họa: ảnh test $256 \times 256$, batch size 1, embed dim 64, 8 attention heads.

---

## 0. Notation

| Ký hiệu | Ý nghĩa | Giá trị mẫu |
|---|---|---|
| `B` | Batch size | 1 |
| `C` | Channel (embed dim) | 64 |
| `H, W` | Height, Width | 256, 256 |
| `h` | Number of attention heads | 8 |
| `d = C/h` | Head dim | 8 |
| `expansion` | FFN expansion factor | 2 |

---

## 1. INPUT — Ảnh gốc (pixel space)

```
I_V  (visible / source): [B, 1, H, W] = [1, 1, 256, 256]   # CT/PET/SPECT_Y
I_I  (infrared / MRI):   [B, 1, H, W] = [1, 1, 256, 256]   # MRI grayscale
```

Cả 2 đều grayscale, normalize về `[0, 1]`. PET/SPECT RGB đã convert Y channel trước khi vào.

---

## 2. ENCODER — Phân rã (Restormer_Encoder, shared cho cả 2 modality)

### 2.1 OverlapPatchEmbed — lifting pixel → feature

```python
self.patch_embed = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
```

```
Input:    [1, 1, 256, 256]   ← ảnh raw 1 kênh
   │
   ▼  Conv 3×3, 1 → 64 channels, stride=1, padding=1
   │
Output:   [1, 64, 256, 256]  ← feature map 64 kênh, giữ nguyên HxW
```

Số tham số: `1 × 64 × 3 × 3 = 576`

### 2.2 SFE (Shallow Feature Extraction) — 4× TransformerBlock

```python
self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=64, num_heads=8, ffn_expansion=2) × 4])
```

Mỗi block giữ nguyên shape:
```
[1, 64, 256, 256]  →  TransformerBlock #1  →  [1, 64, 256, 256]
                   →  TransformerBlock #2  →  [1, 64, 256, 256]
                   →  TransformerBlock #3  →  [1, 64, 256, 256]
                   →  TransformerBlock #4  →  [1, 64, 256, 256]
```

#### Bên trong 1 TransformerBlock

```python
def forward(x):  # x: [1, 64, 256, 256]
    x = x + self.attn(self.norm1(x))     # MDTA channel attention
    x = x + self.ffn(self.norm2(x))      # GDFN feed-forward
    return x
```

#### 2.2.1 MDTA (Multi-Dconv Transposed Attention)

```python
self.qkv = nn.Conv2d(64, 64*3, kernel_size=1)
self.qkv_dwconv = nn.Conv2d(192, 192, kernel_size=3, groups=192)
self.project_out = nn.Conv2d(64, 64, kernel_size=1)
```

```
Input:    [1, 64, 256, 256]
   │
   ▼  Conv 1×1: 64 → 192 (= 3 × 64 cho q, k, v)
   │
   [1, 192, 256, 256]
   │
   ▼  DWConv 3×3 (groups=192)
   │
   [1, 192, 256, 256]
   │
   ▼  Split (chia 3 phần theo channel)
   │
   q [1, 64, 256, 256]   k [1, 64, 256, 256]   v [1, 64, 256, 256]
   │
   ▼  Rearrange thành multi-head:
   │  q: [1, 8 heads, 8 head_dim, 256*256 tokens]
   │  k: [1, 8 heads, 8 head_dim, 256*256 tokens]
   │  v: [1, 8 heads, 8 head_dim, 256*256 tokens]
   │
   ▼  L2 normalize q, k along last dim
   ▼  Attention TRANSPOSE: attn = (q @ k.T) * temperature
   │  shape: [1, 8, 8, 8]   ← chú ý: attention theo CHANNEL, không phải spatial
   │                          (O(C²) thay O((HW)²) — Restormer trick)
   ▼  Softmax theo head_dim
   │
   ▼  out = attn @ v
   │  shape: [1, 8, 8, 65536]
   │
   ▼  Rearrange ngược về spatial
   │
   [1, 64, 256, 256]
   │
   ▼  project_out (Conv 1×1)
   │
Output:   [1, 64, 256, 256]
```

**Key insight**: thay vì attention `[H×W, H×W]` = 65536×65536 (cực lớn), MDTA dùng attention `[C, C]` = 64×64 (nhỏ gọn). Mỗi channel "chú ý" tới các channel khác.

#### 2.2.2 GDFN (Gated-Dconv Feed-Forward Network)

```python
hidden_features = int(64 * 2) = 128
self.project_in = nn.Conv2d(64, 256, kernel_size=1)   # 64 → 128*2 (cho gating)
self.dwconv = nn.Conv2d(256, 256, kernel_size=3, groups=256)
self.project_out = nn.Conv2d(128, 64, kernel_size=1)
```

```
Input:    [1, 64, 256, 256]
   │
   ▼  project_in (Conv 1×1, 64 → 256)
   │
   [1, 256, 256, 256]
   │
   ▼  DWConv 3×3 (groups=256)
   │
   [1, 256, 256, 256]
   │
   ▼  Chunk theo channel thành x1, x2
   │
   x1 [1, 128, 256, 256]   x2 [1, 128, 256, 256]
   │
   ▼  Gating: gelu(x1) * x2
   │
   [1, 128, 256, 256]
   │
   ▼  project_out (Conv 1×1, 128 → 64)
   │
Output:   [1, 64, 256, 256]
```

### 2.3 BaseFeatureExtraction (BTE) — sinh f^B (Base feature)

```python
self.baseFeature = BaseFeatureExtraction(dim=64, num_heads=8)
```

Giống TransformerBlock nhưng dùng `AttentionBase` (spatial MHA chuẩn, không phải MDTA):

```
Input từ SFE:   [1, 64, 256, 256]
   │
   ▼  AttentionBase (spatial multi-head attention)
   │  flatten H*W → tokens, attention [N, N]
   │  N = H*W = 65536  ← lưu ý cost cao hơn MDTA
   │
   ▼  + residual
   ▼  + Mlp(expansion=1, có DWConv)
   │
Output f^B:   [1, 64, 256, 256]
```

### 2.4 DetailFeatureExtraction (DCE) — sinh f^D (Detail feature)

```python
self.detailFeature = DetailFeatureExtraction(num_layers=3)  # 3× DetailNode
```

```python
def forward(x):    # x: [1, 64, 256, 256]
    z1, z2 = x[:, :32], x[:, 32:]      # split theo channel
    for layer in self.net:              # 3× DetailNode
        z1, z2 = layer(z1, z2)
    return torch.cat([z1, z2], dim=1)
```

```
Input:    [1, 64, 256, 256]
   │
   ▼  Split theo channel
   │
   z1 [1, 32, 256, 256]   z2 [1, 32, 256, 256]
   │
   ▼  DetailNode #1 (affine coupling INN)
   │  z1, z2 = layer(z1, z2)
   ▼  DetailNode #2
   ▼  DetailNode #3
   │
   z1 [1, 32, 256, 256]   z2 [1, 32, 256, 256]
   │
   ▼  Concat
   │
Output f^D:   [1, 64, 256, 256]
```

#### Bên trong 1 DetailNode (RealNVP-style)

```python
self.theta_phi = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
self.theta_rho = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
self.theta_eta = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
self.shffleconv = nn.Conv2d(64, 64, kernel_size=1)
```

```python
def forward(z1, z2):                            # z1, z2: [1, 32, 256, 256]
    z = torch.cat([z1, z2], dim=1)              # [1, 64, 256, 256]
    z = self.shffleconv(z)                      # Conv 1×1, mix channels
    z1, z2 = z[:, :32], z[:, 32:]               # split lại
    z2 = z2 + self.theta_phi(z1)                # additive coupling
    z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)  # affine coupling
    return z1, z2
```

Mỗi `theta_*` là InvertedResidualBlock:
```
Input:   [1, 32, 256, 256]
   │
   ▼  PW Conv 1×1: 32 → 64 (expand)
   ▼  ReLU6
   ▼  DW Conv 3×3: 64 → 64 (groups=64)
   ▼  ReLU6
   ▼  PW Conv 1×1: 64 → 32 (project back)
   │
Output:  [1, 32, 256, 256]
```

→ Mỗi DetailNode ~ vài K params, **invertible** (có thể reconstruct ngược).

### 2.5 Output Encoder

Cho mỗi modality, Encoder trả về:
```
f_V_B, f_V_D = Encoder(I_V)       # cả hai [1, 64, 256, 256]
f_I_B, f_I_D = Encoder(I_I)       # cả hai [1, 64, 256, 256]
```

Encoder weights **shared** giữa 2 modality (gọi cùng `model.encoder` 2 lần).

---

## 3. FUSION LAYER — kết hợp 2 modality

### 3a. CDDFuse gốc (paper)

```python
# BaseFuseLayer = BaseFeatureExtraction(dim=64, num_heads=8)  (1 transformer)
# DetailFuseLayer = DetailFeatureExtraction(num_layers=1)     (1 INN block)

f_F^B = BaseFuseLayer(f_I_B + f_V_B)    # phép cộng đơn giản
f_F^D = DetailFuseLayer(f_I_D + f_V_D)
```

```
f_I_B [1, 64, 256, 256] + f_V_B [1, 64, 256, 256]
              │
              ▼  Element-wise add
              │
        [1, 64, 256, 256]
              │
              ▼  BaseFuseLayer (1 Transformer block)
              │
        f_F^B [1, 64, 256, 256]
```

(Tương tự cho Detail với INN.)

### 3b. CDDFuse-AG (đề xuất) — Adaptive Gating

```python
# Adaptive Gating cho Base path
g_B = sigmoid(W_g_B @ concat([f_V_B, f_I_B], dim=1) + b_g_B)
fused_B = g_B * f_V_B + (1 - g_B) * f_I_B
f_F^B = BaseFuseLayer(fused_B)
```

```
f_V_B [1, 64, 256, 256]   f_I_B [1, 64, 256, 256]
              │                      │
              └────────┬─────────────┘
                       │  concat theo channel
                       ▼
                [1, 128, 256, 256]
                       │
                       ▼  Conv 1×1, 128 → 64
                       │
                [1, 64, 256, 256]
                       │
                       ▼  Sigmoid (element-wise)
                       │
                  g_B [1, 64, 256, 256]   ∈ (0, 1)
                       │
                       ▼  Blend: g_B ⊙ f_V_B + (1-g_B) ⊙ f_I_B
                       │
                [1, 64, 256, 256]
                       │
                       ▼  BaseFuseLayer
                       │
                f_F^B [1, 64, 256, 256]
```

→ Số tham số AG-Base: `Conv 1×1 (128→64) + bias = 128 × 64 + 64 = 8,256`. Tổng AG-Base + AG-Detail = `~16.5K params`.

---

## 4. DECODER — Tái tạo ảnh fused

```python
self.reduce_channel = nn.Conv2d(128, 64, kernel_size=1)
self.encoder_level2 = nn.Sequential(*[TransformerBlock × 4])
self.output = nn.Sequential(
    nn.Conv2d(64, 32, kernel_size=3, padding=1),
    nn.LeakyReLU(),
    nn.Conv2d(32, 1, kernel_size=3, padding=1),
)
self.sigmoid = nn.Sigmoid()
```

```
f_F^B [1, 64, 256, 256]   f_F^D [1, 64, 256, 256]
              │                      │
              └────────┬─────────────┘
                       │  concat theo channel
                       ▼
                [1, 128, 256, 256]
                       │
                       ▼  reduce_channel (Conv 1×1, 128 → 64)
                       │
                [1, 64, 256, 256]
                       │
                       ▼  encoder_level2 (4× TransformerBlock)
                       │  (giống SFE của Encoder)
                       │
                [1, 64, 256, 256]
                       │
                       ▼  output head: Conv 3×3 (64 → 32)
                       │
                [1, 32, 256, 256]
                       │
                       ▼  LeakyReLU
                       ▼  Conv 3×3 (32 → 1)
                       │
                [1, 1, 256, 256]
                       │
                       ▼  Sigmoid → range [0, 1]
                       │
Output Î_F:     [1, 1, 256, 256]   ← ảnh tổng hợp
```

---

## 5. POST-PROCESSING (sau Decoder)

```python
out_np = (Î_F.squeeze().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
```

```
Tensor [1, 1, 256, 256] float ∈ [0,1]
       │
       ▼  squeeze
       │
       [256, 256] float
       │
       ▼  × 255, clip [0,255], astype uint8
       │
NumPy [256, 256] uint8 ∈ [0, 255]    ← ảnh grayscale lưu được
```

### Cho PET/SPECT (RGB restoration)

```
Ảnh nguồn RGB (PET)         Ảnh fused Y
       │                         │
       ▼  Convert YCbCr           │
       │                         │
   Y, Cb, Cr                     │
   (drop Y)                      │
       └──────────┬──────────────┘
                  │
                  ▼  Merge thành YCbCr mới
                  │
                  ▼  Convert sang RGB
                  │
        RGB [256, 256, 3] uint8   ← ảnh tổng hợp có màu
```

---

## 6. LOSS COMPUTATION (chỉ khi training)

### 6a. Phase I (epoch 0-39 / 0-14 cho 45ep)

Forward riêng từng modality:
```
I_V → Encoder → (f_V^B, f_V^D) → Decoder → Î_V    # reconstruct V
I_I → Encoder → (f_I^B, f_I^D) → Decoder → Î_I    # reconstruct I
```

```
L_I = α_1 · L_recon(I_V, Î_V) + α_1 · L_recon(I_I, Î_I)
    + α_2 · L_decomp
    + α_3 · L_TV
```

Trong đó `L_decomp = (CC_D)² / (1.01 + CC_B)` với:
- `CC_B = Pearson_corr(f_V^B, f_I^B)` ← scalar
- `CC_D = Pearson_corr(f_V^D, f_I^D)` ← scalar

### 6b. Phase II (epoch 40-119 / 15-44 cho 45ep)

Forward full pipeline:
```
I_V, I_I → Encoder → Fusion (+ AG) → Decoder → Î_F
```

```
L_II = L_fusion + α_4 · L_decomp
L_fusion = L_int^II + L_grad^II
```

#### Paper gốc — max-pixel target
```
L_int^II = ||Î_F - max(I_V, I_I)||_F²
```

Mỗi ảnh `Î_F, I_V, I_I` đều shape `[1, 1, 256, 256]`. `max()` là element-wise:
```
target [1, 1, 256, 256] = element-wise max của (I_V, I_I)
```

#### CDDFuse-AG (Module B) — Saliency-guided target
```
S_V = |∇I_V|     # Sobel gradient magnitude, shape [1, 1, 256, 256]
S_I = |∇I_I|
w = (S_V + ε) / (S_V + S_I + 2ε)   # shape [1, 1, 256, 256], values in (0, 1)
target = w · I_V + (1-w) · I_I      # convex combination

L_int^II = ||Î_F - target||_F²
```

---

## 7. Bảng tổng hợp shape ở mọi điểm chính

| Vị trí | Tensor | Shape | Số params (cumulative) |
|---|---|---|---|
| Input | I_V, I_I | `[1, 1, 256, 256]` | 0 |
| Sau PatchEmbed | x_embed | `[1, 64, 256, 256]` | 576 |
| Sau SFE (4 blocks) | shallow_feat | `[1, 64, 256, 256]` | ~300K |
| Sau BTE | f^B | `[1, 64, 256, 256]` | ~350K |
| Sau DCE (3 INN) | f^D | `[1, 64, 256, 256]` | ~600K |
| **AG output** (Module A) | fused_B / fused_D | `[1, 64, 256, 256]` | +16.5K (AG) |
| Sau BaseFuseLayer | f_F^B | `[1, 64, 256, 256]` | +50K |
| Sau DetailFuseLayer | f_F^D | `[1, 64, 256, 256]` | +30K |
| Sau concat | (in Decoder) | `[1, 128, 256, 256]` | — |
| Sau reduce_channel | x_dec | `[1, 64, 256, 256]` | +8K |
| Sau encoder_level2 | x_refined | `[1, 64, 256, 256]` | +300K |
| Sau output head | logit | `[1, 1, 256, 256]` | +18K |
| Sau sigmoid | Î_F | `[1, 1, 256, 256]` ∈ [0, 1] | — |
| **TOTAL params** | | | **~1.2M** |

---

## 8. Sơ đồ tổng quan (compact)

```
Input pixel space             Feature space (C=64)          Output pixel space
─────────────────             ──────────────────             ──────────────────

I_V [1,1,H,W]
       │
       │  Conv 3×3 (1→64)
       ▼
I_V_embed [1,64,H,W]
       │
       │  SFE: 4× TransformerBlock (MDTA + GDFN)
       ▼
shallow_V [1,64,H,W]
       ├──► BTE (1 Transformer)  ──► f_V^B [1,64,H,W]
       └──► DCE (3 INN blocks)   ──► f_V^D [1,64,H,W]

I_I → same encoder → f_I^B, f_I^D

(AG-Base) ──► g^B = σ(Conv1×1([f_V^B; f_I^B]))   ∈ (0,1)^[1,64,H,W]
              fused_B = g^B ⊙ f_V^B + (1-g^B) ⊙ f_I^B
              ──► BaseFuseLayer ──► f_F^B [1,64,H,W]

(AG-Detail) tương tự ──► f_F^D [1,64,H,W]

f_F^B + f_F^D
       │
       │  concat [1,128,H,W] → Conv 1×1 → [1,64,H,W]
       │  4× TransformerBlock
       │  Conv 3×3 (64→32) → LeakyReLU → Conv 3×3 (32→1) → Sigmoid
       ▼
                                                  Î_F [1,1,H,W] ∈ [0,1]
```

---

## 9. Câu trả lời nhanh các câu hỏi thường gặp

**Q: Tại sao output cũng có sigmoid?**
A: Vì ảnh pixel có range [0, 255] (sau khi × 255), nhưng tensor xử lý normalize về [0, 1]. Sigmoid ép output về [0, 1] cho consistent với input.

**Q: Tại sao AG dùng Conv 1×1 thay Linear?**
A: Conv 1×1 trên feature 4D `[B, 2C, H, W]` tương đương Linear áp dụng độc lập per-pixel. Conv giữ định dạng 4D tiện hơn (không cần reshape).

**Q: Saliency map có cùng shape với ảnh không?**
A: Có. `|∇I_V|` shape `[B, 1, H, W]` giống ảnh nguồn. Weight `w` cũng `[B, 1, H, W]`, broadcast khi multiply.

**Q: Encoder shared weight có nghĩa là gì?**
A: Cùng 1 instance `nn.Module`. Khi call `Encoder(I_V)` rồi `Encoder(I_I)`, gradient cộng dồn vào cùng weights. Khi step optimizer, cả 2 lần forward đều update cùng tham số.

**Q: Tại sao 3 INN block trong Encoder nhưng chỉ 1 INN block trong DetailFuseLayer?**
A: Encoder cần phân rã đủ "deep" (3 blocks) để học decomposition. Fuse layer chỉ refine kết quả đã được decompose, nên 1 block đủ.

**Q: Decoder dùng Transformer khác Encoder không?**
A: Khác cấu trúc nhưng dùng lại `TransformerBlock` (MDTA + GDFN). Cụ thể: Encoder dùng SFE 4 blocks ngay sau patch embed, Decoder dùng encoder_level2 4 blocks sau channel reduction. Cùng loại block, khác vị trí.
