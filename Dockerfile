FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt update && \
    apt install -y bash \
                   build-essential \
                   git \
                   curl \
                   nano \
                   ca-certificates \
                   ffmpeg \
                   libsm6 \
                   libxext6 \
                   python3 \
                   python3-pip && \
    rm -rf /var/lib/apt/lists

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
    torch

RUN git clone https://github.com/NVIDIA/apex
RUN cd apex && \
    python3 setup.py install && \
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

WORKDIR /workspace
COPY . yolov2-pytorch/
RUN cd yolov2-pytorch/

#    && \
#    python3 -m pip install --no-cache-dir .
# CMD ["/bin/bash"]