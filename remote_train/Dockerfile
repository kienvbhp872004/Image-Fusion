# CDDFuse-AG training environment
# Base: CUDA 12.1 runtime + Python 3.8 + PyTorch 2.5.1+cu121
# Hỗ trợ GPU sm_60 (P100), sm_70 (V100), sm_75 (T4), sm_86 (RTX 30xx), sm_89 (RTX 40xx)

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Cài Python 3.10 (Ubuntu 22.04 default) + system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        git wget curl ca-certificates \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# PyTorch 2.5.1 + CUDA 12.1 (hỗ trợ rộng GPU)
RUN pip install --upgrade pip && \
    pip install torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cu121

# Core deps
RUN pip install \
        numpy==1.24.4 \
        scipy \
        scikit-image \
        scikit-learn \
        opencv-python-headless \
        Pillow \
        pandas \
        matplotlib \
        seaborn \
        h5py \
        tqdm \
        einops==0.4.1 \
        kornia==0.6.12 \
        timm \
        tensorboardX

WORKDIR /workspace

# Verify
RUN python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda)"

CMD ["/bin/bash"]
