# With this Dockerfile we create the Docker image with which we run our pipeline
# To create the image, navigate to this folder and
# docker build -t collaborating.tuhh.de:5005/ckv0173/scilab-rl/scilabrl .
# or
# docker buildx build --platform linux/amd64 -t collaborating.tuhh.de:5005/ckv0173/scilab-rl/scilabrl .
# on an ARM device
# To push it to the repo, create a GitLab Personal Access Token and login with
# docker login collaborating.tuhh.de:5005
# then push the image to the repo with
# docker push collaborating.tuhh.de:5005/ckv0173/scilab-rl/scilabrl
# 1. Base image
FROM ubuntu:22.04

# Get necessary system libraries
# adapted from https://github.com/openai/mujoco-py/blob/master/Dockerfile
RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git \
    curl \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    wget \
    python3-pip \
    patchelf \
    libstdc++6 \
    libx11-6  \
    libxcb1  \
    libxau6  \
    xvfb  \
    dbus-x11  \
    x11-utils  \
    libxkbcommon-x11-0 \
    libavcodec-dev  \
    libavformat-dev  \
    libswscale-dev \
    && apt-get autoclean -y && apt-get autoremove -y && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/miniconda3/bin:${PATH}"
