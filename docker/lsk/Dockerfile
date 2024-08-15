# Use an official PyTorch image with CUDA 11.3
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

SHELL ["/bin/bash", "-c"]

# Set the working directory
WORKDIR /workspace
COPY ../../ /workspace/avt-detection/

# TODO: check out commit at mmrotate
RUN <<EOF
    apt-get update --fix-missing && apt-get install -y wget vim git zip unzip build-essential libgl1-mesa-dev ffmpeg libsm6 libxext6
    conda update conda -y
    conda init bash
    conda create -n avt-detection python=3.10 -y
EOF

SHELL ["conda", "run", "-n", "avt-detection", "/bin/bash", "-c"]

RUN <<EOF
    conda install pip=24.0 -y
    conda install conda-forge::ninja=1.12.1
    pip install numpy==1.26.4
    pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
    pip install openmim

    mim install mmcv-full==1.5.3
    mim install mmdet==2.25.2

    git clone --depth 1 --branch main  --single-branch https://github.com/open-mmlab/mmrotate.git 
    cd mmrotate 
    pip install -r requirements/build.txt
    pip install -v -e . 
    mim install mmrotate==0.3.4 
    pip install timm==0.6.13 future tensorboard

    cd  /workspace/avt-detection/
    pip install -r requirements.txt
    cd ./LSKNet && pip install -v -e .
    pip cache purge
    conda clean --all -y
EOF
