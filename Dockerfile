FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -yq --no-install-recommends \
    ca-certificates \
    curl \
    wget \
    git \
    unzip \
    vim \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create and activate Python venv
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Upgrade pip and install PyTorch with CUDA 12.1 support
RUN pip install --upgrade pip
RUN pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# Copy project files
COPY . /app/

# Make launcher scripts executable
RUN chmod +x /app/scripts/*.sh || true

# Install requirements
RUN pip install -r requirements.txt

# Environment variables
ENV LOG_DIR=/app/logs
ENV HF_HOME=/app/dataset
ENV HF_DATASETS_CACHE=/app/dataset
ENV DATASET_BASE_DIR=/app/dataset
ENV HF_HUB_DISABLE_XET=1
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV TOKENIZERS_PARALLELISM=false
ENV PYTHONPATH=/app:$PYTHONPATH

WORKDIR /app
