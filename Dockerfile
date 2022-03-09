FROM nvidia/cuda:11.0-base-ubuntu20.04

ENV LC_ALL="C.UTF-8" LESSCHARSET="utf-8"

WORKDIR /workspace/working


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
    software-properties-common \
    libgl1-mesa-dev

RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt update && apt install -y python3.7 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1
    
COPY ./ ./

RUN make setup

