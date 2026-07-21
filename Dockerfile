# Use the official NVIDIA CUDA base image
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
# FROM nvidia/cuda:11.8.1-cudnn8-devel-ubuntu22.04

# --- 1. System Dependencies ---
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    git \
    wget \
    curl \
    vim \
    build-essential \
    cmake \
    libglew-dev \
    libosmesa6-dev \
    libgl1-mesa-glx \
    patchelf \
    ffmpeg \
    libegl1 \
    python3-venv \
    ninja-build \
    && apt-get clean

# RUN apt-get install -y python3-venv

# Make python3.10 the default 'python3' and 'python'
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1


RUN python3.10 -m venv /opt/venv
# Set the PATH to use the venv's binaries
ENV PATH="/opt/venv/bin:$PATH"
ENV HOME=/workspace/DisentangledFlow
# --- 2. Python Core Dependencies ---
# Set a general workspace
WORKDIR ${HOME}

COPY . . 

# DO NOT upgrade pip globally. Use the system-provided one which is tied to python3.10
# Use python3.10 -m pip to be 100% explicit
RUN python3.10 -m pip install --upgrade pip setuptools wheel
RUN python3.10 -m pip install numpy==1.24
RUN python3.10 -m pip install --upgrade "jax[cuda11_pip]==0.4.20" "jaxlib==0.4.20+cuda11.cudnn86" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN python3.10 -m pip install scipy==1.10
RUN python3.10 -m pip install mujoco

# --- 3. Install Projects ---
# --- LIBERO ---
WORKDIR $HOME/benchmarks
RUN git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
WORKDIR $HOME/benchmarks/LIBERO
RUN python3.10 -m pip install -r requirements.txt
RUN python3.10 -m pip install -e . --config-settings editable_mode=compat

# --- Octo (Your project) ---
WORKDIR $HOME/vlas
RUN git clone https://github.com/octo-models/octo.git
WORKDIR $HOME/vlas/octo
RUN python3.10 -m pip install -r requirements.txt
RUN python3.10 -m pip install -e .

# --- ACT ---
WORKDIR /workspace
# RUN git clone https://github.com/tonyzhaozh/act
# ACT requirements are already installed by Octo/Libero, but we install dm_control just in case
RUN python3.10 -m pip install dm_control ipython

# --- OpenVLA-OFT ---
WORKDIR $HOME/vlas
RUN git clone https://github.com/moojink/openvla-oft.git
WORKDIR $HOME/vlas/openvla_oft
RUN python3.10 -m pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 orbax-checkpoint==0.4.0 draccus
RUN python3.10 -m pip install numpy==1.24.1 scipy==1.10.0 diffusers==0.30.3 json-numpy==2.1.1 timm==0.9.10 jsonlines wandb==0.13.1 einops==0.8.1 peft==0.11.1 gym==0.26.2 robosuite==1.4.0 egl_probe==1.0.2 huggingface-hub==0.36.0 pillow==12.0.0
# Flash-Attention is strictly required for fast training on OpenVLA and OFT.
RUN python3.10 -m pip install packaging ninja
RUN python3.10 -m pip install flash-attn==2.5.6 --no-build-isolation
RUN python3.10 -m pip install git+https://github.com/moojink/transformers-openvla-oft.git@bc339d9ad707454c0c115970db43c260067c61ab
RUN python3.10 -m pip install -e .

# --- 4. Final Config ---
# Set default directory back to your project
WORKDIR $HOME

RUN git config --global --add safe.directory $HOME/octo
RUN git config --global --add safe.directory $HOME/openvla-oft

ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV MUJOCO_GL=egl

# Re-pin jaxlib to the CUDA 12.1-compatible build (cudnn88).
# Later installs (orbax, transformers, openvla-oft) tend to upgrade jaxlib
# to cudnn89 which requires CUDA 12.2+ and breaks GPU detection.
RUN python3.10 -m pip install --force-reinstall \
    "jaxlib==0.4.20+cuda11.cudnn86" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

CMD ["bash"]


