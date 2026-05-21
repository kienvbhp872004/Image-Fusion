# Docker run command — copy/paste ready

> Tổng hợp các lệnh sẵn dùng cho bạn (người chạy train).

## 1. Verify GPU passthrough

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

Phải in ra bảng GPU info. Nếu không → cài `nvidia-container-toolkit`:
```bash
# Ubuntu / Debian
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## 2. Build image

```bash
cd /path/to/cddfuse_ag    # nơi giải nén package
docker build -t cddfuse-ag:latest -f remote_train/Dockerfile remote_train/
```

## 3. Train (one-shot, recommended)

```bash
docker run --rm --gpus all \
    --shm-size=8g \
    -v $(pwd):/workspace \
    -w /workspace \
    cddfuse-ag:latest \
    bash /workspace/remote_train/run_train.sh
```

- `--shm-size=8g`: DataLoader cần shared memory cho multi-worker.
- `-v $(pwd):/workspace`: mount thư mục hiện tại → output sẽ ở `./output/`.

## 4. Interactive shell (debug)

Nếu muốn vào container thử lệnh:
```bash
docker run -it --rm --gpus all \
    --shm-size=8g \
    -v $(pwd):/workspace \
    -w /workspace \
    cddfuse-ag:latest \
    bash
```

Trong container:
```bash
# Verify GPU + torch
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Run từng bước
cd /workspace/models/MMIF-CDDFuse
python dataprocessing_MIF.py
python train_MIF.py --variant Combined-Gated-Saliency --batch 16 --num_epochs 120 --epoch_gap 40 --coeff_tv 10.0 --amp --output /workspace/output/
```

## 5. Background + log

Nếu muốn để chạy nền, tail log:
```bash
docker run -d --gpus all \
    --shm-size=8g \
    --name cddfuse-train \
    -v $(pwd):/workspace \
    -w /workspace \
    cddfuse-ag:latest \
    bash /workspace/remote_train/run_train.sh

# Theo dõi log
docker logs -f cddfuse-train

# Khi xong
docker rm cddfuse-train
```

## 6. Multi-GPU (nếu có nhiều A100)

`train_MIF.py` hiện chỉ single-GPU. Nếu bạn muốn nhanh hơn:
```bash
# Chọn 1 GPU cụ thể
docker run --rm --gpus '"device=0"' ...
```
Multi-GPU DDP cần refactor code — tạm thời chỉ dùng 1 GPU.

## 7. Resume training (nếu bị disconnect)

Hiện tại chưa hỗ trợ resume. Nếu container chết giữa chừng, phải chạy lại từ epoch 0. Khuyến cáo: chạy với `tmux` hoặc `nohup` để khỏi bị mất kết nối SSH.

```bash
tmux new -s train
docker run --rm --gpus all ...   # paste lệnh
# Ctrl+B then D để detach
# tmux attach -t train  để xem lại
```
