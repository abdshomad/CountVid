# Builder stage
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc-11 g++-11 \
        wget \
        git \
        python3.10 \
        python3.10-venv \
        python3.10-distutils \
        python3-pip \
        ca-certificates \
        && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    python3 -m pip install --upgrade pip

RUN ln -s /usr/bin/python3 /usr/bin/python \
    || true

WORKDIR /workspace

COPY . /workspace

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Build and install GroundingDINO ops
RUN cd models/GroundingDINO/ops && \
    python setup.py build install && \
    python test.py

RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'
RUN pip install gdown

# Download BERT weights and checkpoints
RUN python download_bert.py
RUN mkdir -p checkpoints
RUN gdown --id 1bw-YIS-Il5efGgUqGVisIZ8ekrhhf_FD -O checkpoints/countgd_box.pth
RUN wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# Final stage
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-venv \
        python3.10-distutils \
        python3-pip \
        ca-certificates \
        wget \
        git \
        && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    python3 -m pip install --upgrade pip
RUN ln -s /usr/bin/python3 /usr/bin/python \
    || true

WORKDIR /workspace

# Copy installed site-packages and binaries from builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project files and checkpoints from builder
COPY --from=builder /workspace /workspace

# Set default command
CMD ["/bin/bash"] 