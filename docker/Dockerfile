#
# FILE: Dockerfile
#
# DESCRIPTION: Docker image for NeRF experiments using NerfStudio or SdfStudio including Tiny CUDA NN framework
#
# AUTHOR: Deeepwin
#
# DATE: 04.08.2023
#
# INSTALLATION: 
#
#   Ensure correct host installation to support CUDA container runtime. Follow these instructions:
# 
#   https://collabnix.com/introducing-new-docker-cli-api-support-for-nvidia-gpus-under-docker-engine-19-03-0-beta-release/
#
# USAGE: 
#
#   To create docker image:
#
#   1. Go to folder where dockerfile is located
#   2. Run `docker build --progress plain -t <image name> .`
#
#   To run docker image with GPU support:
#
#   docker run -it --rm --gpus all <docker tag>
#
#   To free up disk space, remove dangling images:
#
#   docker builder prune && docker system prune
#

# includes CUDA 11.3.1, cuDNN 8.2.0.53
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# set timezone
ENV TZ=Europe/Zurich

# force non-interactive to bypass prompts in installation
ARG DEBIAN_FRONTEND=noninteractive

# for installation use bash, shell will not work well
SHELL ["/bin/bash", "-c"]

# create user, but do not yet activate
RUN useradd -rm -d /home/ipca -s /bin/bash -g root -G sudo -u 1001 ipca
WORKDIR /home/ipca

# install python environment
RUN apt-get update \
    && apt-get -y install python3.8 \
    && apt-get -y install python3.8-venv \
    && apt-get -y install python3-opencv \
    && apt-get -y install python3-dev \
    && apt-get -y install git \
    && apt-get -y install curl \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# install c compiler version 9
RUN apt-get -y install g++-9 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 9 \
    && update-alternatives --config g++ \
    && apt-get -y install gcc-9 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 9 \
    && update-alternatives --config gcc

# install cuda toolkit, check with nvcc --version
RUN apt-get -y install nvidia-cuda-toolkit

# setup environment variables (docker image configuration, not host)
ENV CUDA_HOME /usr/local/cuda-11.3
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64
ENV TCNN_CUDA_ARCHITECTURES 75

# setup virtual environments
RUN mkdir envs \
    && python3 -m venv envs/sdfstudio \
    && python3 -m venv envs/nerfstudio

# install pytorch into `sdfstudio`
RUN source envs/sdfstudio/bin/activate \
    && pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# install tinycudann
RUN source envs/sdfstudio/bin/activate \
    && pip install wheel \
    && pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# install nerfstudio and sdfstudio
RUN source envs/sdfstudio/bin/activate \
    && python3 -m pip install --upgrade pip \
    && pip install nerfstudio \
    && mkdir nerf \
    && cd nerf \
    && git clone https://github.com/autonomousvision/sdfstudio.git \
    && cd sdfstudio \
    && pip install -e . \
    && cd .. \
    && cd ..

# ensure compatibility
RUN source envs/sdfstudio/bin/activate \
  && pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html \
  && pip install nerfacc -f https://nerfacc-bucket.s3.us-west-2.amazonaws.com/whl/torch-1.12.1_cu113.html \
  && pip install Pillow==9.5.0 \
  && pip install functorch==0.2.1

# clean up
RUN source envs/sdfstudio/bin/activate \
    && apt-get clean \
    && apt-get autoclean

# activate user
RUN chown -R ipca /home/ipca/nerf
USER ipca

# finish
#CMD source envs/sdfstudio/bin/activate && ns-install-cli --mode install && /bin/bash

