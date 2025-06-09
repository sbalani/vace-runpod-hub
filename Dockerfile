# base image with cuda 12.1
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# install python 3.11 and pip
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    gnupg2 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
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

# set python3.11 as the default python
RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python3

# install uv
RUN pip install uv

# create venv
ENV PATH="/.venv/bin:${PATH}"
RUN uv venv --python 3.11 /.venv

# install dependencies
RUN uv pip install torch --extra-index-url https://download.pytorch.org/whl/cu121

# install remaining dependencies from PyPI
COPY requirements.txt /requirements.txt
RUN uv pip install -r /requirements.txt

# Add Tini (statically-linked version)
ENV TINI_VERSION v0.19.0
ADD https://github.com/krallin/tini/releases/download/v0.19.0/tini-static /tini
RUN chmod +x /tini

# Create working directory
WORKDIR /workspace

# Copy the VACE code
COPY . /workspace/

# Set the entrypoint for RunPod with Tini (with debug flags)
ENTRYPOINT ["/tini", "-vvv", "--", "python", "-u", "runpod_handler.py"] 