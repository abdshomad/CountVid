# Use NVIDIA CUDA base image with Python 3.10
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
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

# Set gcc/g++ alternatives to 11
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100

# Set python3 and pip3 to python3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    python3 -m pip install --upgrade pip

# Add symlink for python -> python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /workspace

# Copy the project files
COPY . /workspace

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Build and install GroundingDINO ops
RUN cd models/GroundingDINO/ops && \
    python setup.py build install && \
    python test.py

# Install detectron2 from GitHub
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Download BERT weights
RUN python download_bert.py

# Install gdown for downloading from Google Drive
RUN pip install gdown

# Create checkpoints directory if it doesn't exist
RUN mkdir -p checkpoints

# Download the pretrained CountGD-Box model
RUN gdown --id 1bw-YIS-Il5efGgUqGVisIZ8ekrhhf_FD -O checkpoints/countgd_box.pth

# Download the pretrained SAM 2.1 weights
RUN wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# Set default command
CMD ["/bin/bash"] 