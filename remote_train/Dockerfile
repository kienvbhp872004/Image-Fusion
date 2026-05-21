# CDDFuse-AG training environment
# Base: CUDA 12.1 + Ubuntu 20.04 (default Python 3.8) + PyTorch 2.4.1+cu121
# Hỗ trợ GPU sm_70+ (V100, T4, A100, A30, L40, L40s, RTX 30xx/40xx)
# Python 3.8 để khớp project gốc (paper CDDFuse phát triển trên py3.8 + kornia 0.6).

FROM nvidia/cuda:12.1.1-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Ubuntu 20.04 default Python = 3.8
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        git wget curl ca-certificates unzip \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && python --version | grep -q "3.8"

# PyTorch 2.4.1 — bản cuối hỗ trợ Python 3.8, vẫn dùng CUDA 12.1 (sm_70+)
RUN pip install --upgrade pip && \
    pip install torch==2.4.1 torchvision==0.19.1 \
        --index-url https://download.pytorch.org/whl/cu121

# Core deps (versions khớp project)
RUN pip install \
        numpy==1.24.4 \
        scipy \
        scikit-image==0.19.2 \
        scikit-learn \
        opencv-python-headless==4.5.3.56 \
        Pillow \
        pandas \
        matplotlib \
        seaborn \
        h5py \
        tqdm \
        einops==0.4.1 \
        kornia==0.6.12 \
        timm==0.4.12 \
        tensorboardX==2.5.1

WORKDIR /workspace

# Verify
RUN python -c "import torch; print('python:', __import__('sys').version.split()[0]); print('torch:', torch.__version__, 'cuda:', torch.version.cuda)"

CMD ["/bin/bash"]
