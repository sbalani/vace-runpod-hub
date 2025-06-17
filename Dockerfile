# base image with cuda 12.1
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# Create working directory
WORKDIR /workspace

# install python 3.11 and pip
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    gnupg2 \
    python3-apt \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment and install dependencies
RUN python3.11 -m venv venv && \
    . /workspace/venv/bin/activate && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 && \
    pip install decord ninja wheel packaging onnxruntime-gpu && \
    pip uninstall -y flash_attn

# Clone and install flash-attention
RUN cd /workspace && \
    git clone https://github.com/Dao-AILab/flash-attention && \
    cd flash-attention && \
    MAX_JOBS=2 python setup.py install

# Clone VACE and install dependencies
RUN cd /workspace && \
    git clone https://github.com/pftq/VACE_Fixes && \
    cd VACE_Fixes && \
    . /workspace/venv/bin/activate && \
    pip install -r requirements.txt && \
    pip install wan@git+https://github.com/Wan-Video/Wan2.1 && \
    pip install ltx-video@git+https://github.com/Lightricks/LTX-Video@ltx-video-0.9.1 sentencepiece --no-deps && \
    pip install "huggingface_hub[cli]" && \
    huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./models/Wan2.1-T2V-14B && \
    huggingface-cli download ali-vilab/VACE-Wan2.1-14B-Preview --local-dir ./models/VACE-Wan2.1-14B && \
    huggingface-cli download ali-vilab/VACE-Benchmark --repo-type dataset --local-dir ./benchmarks/VACE-Benchmark && \
    pip install -r requirements/annotator.txt && \
    pip install xfuser

# Install GroundingDINO
RUN cd /workspace && \
    git clone https://github.com/pftq/GroundingDINO_Fix && \
    cd GroundingDINO_Fix && \
    . /workspace/venv/bin/activate && \
    pip install .

# Set the entrypoint for RunPod with Tini (with debug flags)
CMD ["python", "-u", "runpod_handler.py"]