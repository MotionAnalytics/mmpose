# ------------------------------------------------------------
# Base image – modern PyTorch with CUDA 11.8 + cuDNN 8
# ------------------------------------------------------------

# Downgrade PYTORCH to 2.1.0 (Best compatibility with MMCV wheels)
ARG PYTORCH="2.1.0" 
ARG CUDA="11.8"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# ------------------------------------------------------------
# Environment setup
# ------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV FORCE_CUDA="1"

# Set PYTHONPATH explicitly
ENV PYTHONPATH="/opt/project"

# ------------------------------------------------------------
# Fix for NVIDIA repo key & basic dependencies
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Install Python deps and OpenMMLab stack
# ------------------------------------------------------------
RUN pip install --upgrade pip setuptools wheel
# Fix Numpy version early
RUN pip install cython openmim "numpy<2.0.0"

# Install xtcocoapi without build isolation
RUN pip install --no-build-isolation git+https://github.com/jin-s13/xtcocoapi

# Install MMEngine
RUN pip install "mmengine>=0.4.0,<1.0.0"

# Install MMCV using MIM (Torch 2.1.0 HAS official wheels, so this will work correctly)
RUN mim install "mmcv>=2.1.0,<2.2.0"

# ------------------------------------------------------------
# Clone and install MMPose
# ------------------------------------------------------------
RUN git clone https://github.com/open-mmlab/mmpose.git /mmpose
WORKDIR /mmpose

# Pin to specific commit for reproducibility (replace with actual commit if needed)
RUN git checkout main

RUN pip install -r requirements/build.txt

# Pre-install chumpy without isolation
RUN pip install --no-build-isolation chumpy

# Install mmpose in editable mode
RUN pip install --no-cache-dir -e .
RUN pip install notebook

# ------------------------------------------------------------
# Install MMDetection & fix xtcocotools incompatibility
# ------------------------------------------------------------
RUN mim install "mmdet>=3.0.0,<3.3.0"


# Uninstall the pre-compiled xtcocotools and reinstall it from source
# to ensure it's compiled against the correct NumPy version.
RUN pip uninstall xtcocotools -y
RUN pip install --no-build-isolation --no-binary xtcocotools xtcocotools

RUN mim install "mmpretrain"
RUN mim install "mmtrack"
RUN command pip install "matplotlib<3.10"

# ------------------------------------------------------------
# Clean up
# ------------------------------------------------------------
RUN conda clean --all -f -y

# ------------------------------------------------------------
# Make conda activate work out-of-the-box in interactive shells
# ------------------------------------------------------------
SHELL ["/bin/bash", "--login", "-c"]

RUN conda init bash && echo "conda activate base" >> ~/.bashrc

WORKDIR /opt/project