FROM nvidia/cuda:11.0-base-ubuntu20.04

ENV LC_ALL="C.UTF-8" LESSCHARSET="utf-8"

WORKDIR /workspace/working


# nvidia-dockerのGPGキーが更新されたから
# Ref: https://github.com/NVIDIA/nvidia-docker/issues/1631
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-get update && apt-get upgrade -y \
    && DEBIAN_FRONTEND=nointeractivetzdata \
    TZ=Asia/Tokyo \
    apt-get install -y \
    zsh \
    python-is-python3 \
    make \
    git \
    python3.7 \
    python3-pip \
    python3-distutils \
    python3-apt \
    software-properties-common \
    libgl1-mesa-dev

RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt update && apt install -y python3.7 \
    && apt install -y python3.7-distutils \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1
    
COPY ./ ./

RUN make setup

