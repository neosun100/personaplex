# PersonaPlex All-in-One Docker Image
ARG BASE_IMAGE="nvidia/cuda"
ARG BASE_IMAGE_TAG="12.4.1-cudnn-runtime-ubuntu22.04"

FROM ${BASE_IMAGE}:${BASE_IMAGE_TAG}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libopus-dev \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip and setuptools first (Ubuntu 22.04 ships ancient versions)
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

# Install PyTorch first
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Copy moshi package and install with all dependencies
COPY moshi/ /app/moshi/
RUN pip3 install --no-cache-dir /app/moshi/ && \
    python3 -c "import sphn, sentencepiece, aiohttp; print('moshi deps OK')"

# Install additional dependencies
RUN pip3 install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    aiofiles \
    jinja2 \
    fastmcp

# Copy application files
COPY app/ /app/app/
COPY assets/ /app/assets/

# Create directories
RUN mkdir -p /app/ssl /tmp/personaplex /app/outputs

EXPOSE 8998

ENV PORT=8998
ENV HOST=0.0.0.0
ENV DEVICE=cuda
ENV CPU_OFFLOAD=false

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["python3", "-m", "app.server"]
