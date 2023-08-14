#
# FILE: Dockerfile
#
# DESCRIPTION: Docker image for NeRF experiments (multinerf) using JAX
#
# AUTHOR: Deeepwin
#
# DATE: 11.08.2023
#
# INSTALLATION: 
#
#   Ensure correct host installation to support CUDA container runtime. Follow these instructions:
# 
#   https://collabnix.com/introducing-new-docker-cli-api-support-for-nvidia-gpus-under-docker-engine-19-03-0-beta-release/
#
#
# IMPORTANCE NOTICE:
#
#   - Keep docker layer size below 5GB, otherwise singularity/apptainer conversion won't work
#
# USAGE: 
#
#   To create docker image:
#
#       1. Go to folder where dockerfile is located
#       2. Run `docker build --progress plain -t <image name> .`
#
#   To run docker image with GPU support:
#
#       docker run -it --rm --gpus all <docker tag>
#
#   To free up disk space, remove dangling images:
#
#       docker builder prune && docker system prune
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

# do not create separate user account, as singularity would use host account
WORKDIR /home/env

# install python environment
RUN apt-get update \
    && apt-get -y install python3.8 \
    && apt-get -y install python3.8-venv \
    && apt-get -y install python3-opencv \
    && apt-get -y install python3-dev \
    && apt-get -y install git \
    && apt-get -y install curl \
    && apt-get -y install ffmpeg \
    && apt-get clean \
    && apt-get autoclean \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# setup virtual environments
RUN mkdir envs \
    && python3 -m venv envs/jax

# install pytorch into `jax` environment
RUN source envs/jax/bin/activate \
    && pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html --no-cache-dir

# install jax
RUN source envs/jax/bin/activate \
  && pip install pip install https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.4.7+cuda11.cudnn82-cp38-cp38-manylinux2014_x86_64.whl --no-cache-dir

# install other pip packes
RUN source envs/jax/bin/activate \
    && python3 -m pip install --upgrade pip \
    && pip install absl-py \
    && pip install flax==0.5.2 \
    && pip install tensorflow \
    && pip install gin-config \
    && pip install opencv-python \
    && pip install dm-pix \
    && pip install rawpy \
    && pip install mediapy \
    && pip install ffmpeg
    
# finish
CMD source /home/env/envs/jax/bin/activate \
    && /bin/bash