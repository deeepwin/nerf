#
# FILE: Dockerfile
#
# DESCRIPTION: Docker image for Neuralangelo but with CUDA 12.0 instead of 12.1
#
# AUTHOR: Deeepwin
#
# DATE: 19.08.2023
#
# USAGE: 
#
#   docker build -t nerf-neuralangelo-120 .
#
# SOURCE:
#
# Original code: https://github.com/NVlabs/neuralangelo/blob/main/docker/Dockerfile-neuralangelo
#
FROM nvcr.io/nvidia/pytorch:23.02-py3
ARG DEBIAN_FRONTEND=noninteractive

# Install basics
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    bzip2 \
    ca-certificates \
    cmake \
    curl \
    ffmpeg \
    g++ \
    git \
    libx264-dev \
    tmux \
    wget

# Update pip
RUN pip install --upgrade pip

# Code formatting
RUN pip install --upgrade \
    flake8 \
    pre-commit

# Install base Python libraries for Imaginaire
COPY requirements.txt requirements.txt
ARG FORCE_CUDA=1
ARG TCNN_CUDA_ARCHITECTURES=70,72,75,80,86
RUN pip install --upgrade -r requirements.txt