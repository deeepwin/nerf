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
#
# IMPORTANCE NOTICE:
#
#   Keep docker layer size below 5GB, otherwise singularity/apptainer conversion won't work
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
    && apt-get clean \
    && apt-get autoclean \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# install nvidia environment, check with nvcc --version
RUN apt-get -y install nvidia-utils-525 \ 
    && apt-get -y install nvidia-cuda-toolkit \
    && apt-get clean \
    && apt-get autoclean

# setup environment variables (docker image configuration, not host)
ENV CUDA_HOME /usr/local/cuda-11.3
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64
ENV TCNN_CUDA_ARCHITECTURES 75

# setup virtual environments
RUN mkdir envs \
    && python3 -m venv envs/sdfstudio \
    && python3 -m venv envs/nerfstudio

# install pytorch into `sdfstudio` environment
RUN source envs/sdfstudio/bin/activate \
    && pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html --no-cache-dir

# install tinycudann
RUN source envs/sdfstudio/bin/activate \
    && pip install wheel --no-cache-dir \
    && pip install ninja --no-cache-dir

RUN source envs/sdfstudio/bin/activate \
  && pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch --no-cache-dir

# install nerfstudio, use nerfstudio 0.1.9 to keep torch 1.12.1, don't upgrade otherwise layer will blow up 
RUN source envs/sdfstudio/bin/activate \
    && python3 -m pip install --upgrade pip \
    && pip install --upgrade-strategy=only-if-needed nerfstudio==0.1.9 --no-cache-dir

# install sdfstudio
RUN source envs/sdfstudio/bin/activate \
    && mkdir tools \
    && cd tools \
    && git clone https://github.com/autonomousvision/sdfstudio.git \
    && cd sdfstudio \
    && pip install -e . --no-cache-dir \
    && cd .. \
    && cd ..

# ensure compatibility, this should not change the installation, cross check
RUN source envs/sdfstudio/bin/activate \
  && pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html --no-cache-dir \
  && pip install nerfacc -f https://nerfacc-bucket.s3.us-west-2.amazonaws.com/whl/torch-1.12.1_cu113.html --no-cache-dir \
  && pip install Pillow==9.5.0 --no-cache-dir \
  && pip install functorch==0.2.1 --no-cache-dir
    
# finish
CMD source /home/env/envs/sdfstudio/bin/activate \
#     && ns-install-cli --mode install \
    && /bin/bash