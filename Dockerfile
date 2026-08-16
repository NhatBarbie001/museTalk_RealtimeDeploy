# ==========================================
# MuseTalk Live Avatar GPU Production Image
# Base: Ubuntu 22.04 with CUDA 11.8 & cuDNN 8
# ==========================================
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

WORKDIR /workspace

# Install system dependencies & build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch 2.0.1 with CUDA 11.8
RUN pip install --no-cache-dir \
    torch==2.0.1 \
    torchvision==0.15.2 \
    torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

# Install OpenMMLab packages with official prebuilt wheels
RUN pip install --no-cache-dir mmengine "mmdet==3.1.0" "mmpose==1.1.0" && \
    pip install --no-cache-dir "mmcv==2.0.1" -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install WebRTC, Server and Streaming dependencies
RUN pip install --no-cache-dir \
    fastapi==0.110.0 \
    uvicorn[standard]==0.28.0 \
    aiortc==1.8.0 \
    av==11.0.0 \
    python-multipart==0.0.9 \
    pydantic==2.6.4 \
    huggingface_hub

# Copy application source code
COPY . /workspace

# Make entrypoint executable
RUN chmod +x /workspace/entrypoint.sh /workspace/download_weights.sh

EXPOSE 8000
EXPOSE 10000-20000/udp

ENTRYPOINT ["/workspace/entrypoint.sh"]
