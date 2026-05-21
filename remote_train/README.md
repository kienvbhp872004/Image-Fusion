# Hướng dẫn train CDDFuse-AG trên GPU remote (Docker)

> Package này dùng để train mô hình **CDDFuse-AG** (Adaptive Gating + Saliency-guided Pixel) cho bài toán tổng hợp ảnh y tế. Train 120 epoch, batch 16, ~3-5h trên A100/A30/L40/L40s.

---

## 1. Yêu cầu phía bạn (người chạy)

### Hardware
- GPU NVIDIA Ampere/Ada (sm_80+): **A100, A30, L40, L40s** — đều OK.
- ≥ 24 GB VRAM (batch 16 cần ~20 GB peak với AMP fp16).
- ≥ 30 GB disk free.

### Software
- Docker (≥ 20.10) + **NVIDIA Container Toolkit** (`nvidia-docker2`).
- Kiểm tra GPU passthrough hoạt động:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```
Phải thấy GPU output. Nếu báo `unknown runtime: nvidia` → cài thêm `nvidia-container-toolkit`.

---

## 2. Lấy package

Mình gửi file zip qua Google Drive: **[link Drive sẽ điền sau]**

```bash
# Tải về và giải nén
wget -O cddfuse_ag_package.zip "<link Drive>"
unzip cddfuse_ag_package.zip -d cddfuse_ag
cd cddfuse_ag
ls -la
# Phải thấy: Dockerfile, run_train.sh, models/, metric/, data/, Havard-Medical-Image-Fusion-Datasets-main/, README.md
```

---

## 3. Build Docker image

```bash
cd cddfuse_ag/remote_train      # hoặc nơi có Dockerfile
docker build -t cddfuse-ag:latest -f Dockerfile .
```

Thời gian build: ~5-10 phút (tải PyTorch + deps).

Kiểm tra image:
```bash
docker images | grep cddfuse-ag
# cddfuse-ag   latest   <hash>   ...
```

---

## 4. Chạy training

```bash
# Từ thư mục cddfuse_ag (root của package, chứa models/, data/, ...)
cd cddfuse_ag

docker run --rm --gpus all \
    -v $(pwd):/workspace \
    -w /workspace \
    cddfuse-ag:latest \
    bash /workspace/remote_train/run_train.sh
```

### Giải thích các flag

| Flag | Ý nghĩa |
|---|---|
| `--rm` | Xóa container sau khi xong (giữ output qua volume) |
| `--gpus all` | Cho phép container dùng GPU |
| `-v $(pwd):/workspace` | Mount thư mục hiện tại vào `/workspace` trong container |
| `-w /workspace` | Set working directory |
| `bash /workspace/remote_train/run_train.sh` | Script chạy 3 bước: preprocess → train → evaluate |

### Tiến trình

Script in log tqdm theo từng epoch. Bạn sẽ thấy:

```
[1/3] Preprocess: extract patches → HDF5
extract patches: 100%|████| 738/738 [00:15<00:00]
[h5  ] ~6500 patches saved

[2/3] Train CDDFuse-AG (120 ep, batch 16, α₃=10, AMP fp16)
ep001/120 P1: 100%|████| 407/407 [01:30<00:00]  loss=0.5234
ep002/120 P1: ...
...
ep040/120 P1: ...
ep041/120 P2: ...
...
ep120/120 P2: ...

[3/3] Evaluate trên 72 cặp test
... (eval per modal)

=== HOÀN TẤT ===
Checkpoint: /workspace/output/CDDFuse-Combined-Gated-Saliency_MIF_*.pth
Metrics:    /workspace/output/eval/
```

### Thời gian dự kiến

| GPU | Thời gian total |
|---|---|
| A100 80GB | ~2-3h |
| A100 40GB | ~3h |
| A30 24GB | ~4-5h |
| L40 / L40s 48GB | ~2.5-3h |

---

## 5. Lấy kết quả gửi lại

Sau khi xong, các file quan trọng nằm ở `cddfuse_ag/output/`:

| File | Vai trò |
|---|---|
| `CDDFuse-Combined-Gated-Saliency_MIF_*.pth` | Checkpoint chính (~5 MB) |
| `CDDFuse-Combined-Gated-Saliency_MIF_*_train_history.json` | Loss curves theo epoch |
| `eval/perimage/CDDFuse-Combined-Gated-Saliency_{CT,PET,SPECT}_perimage.csv` | Per-image metrics (input cho stats) |
| `eval/CDDFuse-Combined-Gated-Saliency_summary.csv` | Aggregate metrics 3 modality |
| `eval/Fusion/*.png` | Ảnh fused output (72 ảnh) |

Đóng gói gửi mình:
```bash
zip -r output_ag_120ep.zip cddfuse_ag/output/
# Upload lên Drive, gửi link
```

Tổng size output: ~20-30 MB.

---

## 6. Troubleshooting

### `Failed to initialize NVML: Unknown Error`
Container không thấy GPU. Kiểm tra:
```bash
docker info | grep -i nvidia
# Phải thấy: Runtimes: nvidia runc
```
Nếu không có, cài `nvidia-container-toolkit` rồi `sudo systemctl restart docker`.

### `RuntimeError: CUDA out of memory` (batch 16)
Hiếm với 24+GB VRAM, nhưng nếu xảy ra:
```bash
# Edit run_train.sh, đổi --batch 16 thành --batch 8
docker run ... bash -c "sed -i 's/--batch       16/--batch       8/' /workspace/remote_train/run_train.sh && bash /workspace/remote_train/run_train.sh"
```

### `ModuleNotFoundError: No module named 'X'`
Container thiếu deps (không nên xảy ra). Rebuild:
```bash
docker build --no-cache -t cddfuse-ag:latest -f Dockerfile .
```

### Training quá chậm (vd > 5 phút/epoch)
- Kiểm tra `nvidia-smi` xem GPU có dùng 100% không.
- Có thể do data loading bottleneck. Edit `train_MIF.py`, set `num_workers=4` trong DataLoader.

---

## 7. Cấu hình thí nghiệm (FYI)

Theo paper CDDFuse (CVPR 2023) §5.1:

| Hyperparameter | Value | Note |
|---|---|---|
| Epochs total | 120 | Phase I = 40, Phase II = 80 |
| Batch size | 16 | Paper text §5.1 |
| Learning rate | 1e-4 | StepLR γ=0.5 mỗi 20 ep, floor 1e-6 |
| Optimizer | Adam | 6 cái riêng (Enc, Dec, BaseFuse, DetailFuse, AG-Base, AG-Detail) |
| Clip grad | 0.01 | |
| α₁ (MSE) | 1 | recon loss |
| α₂ (decomp) | 2 | Phase I |
| **α₃ (TV)** | **10** | paper text (code release = 5) |
| α₄ (decomp II) | 2 | Phase II |
| Patch size | 128×128 | stride 64 |
| Mixed precision | fp16 (AMP) | |
| Seed | 42 | |

Dataset:
- Train: **738 cặp** (160 CT-MRI + 245 PET-MRI + 333 SPECT-MRI)
- Test: **72 cặp** (24 mỗi modality)

---

## 8. Liên hệ

Có vấn đề gì gọi/nhắn Đỗ Trung Kiên — kien.dt224869@sis.hust.edu.vn.

Cảm ơn bạn nhiều! 🙏
