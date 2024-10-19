# Use an official PyTorch image with CUDA 11.3
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

SHELL ["/bin/bash", "-c"]

# Set the working directory
WORKDIR /workspace
COPY ../../ /workspace/avt-detection/

# TODO: check out commit at mmrotate
RUN <<EOF
    apt-get update --fix-missing && apt-get install -y wget vim git zip unzip build-essential libgl1-mesa-dev ffmpeg libsm6 libxext6 gdal-bin
    conda update conda -y
    conda init bash
    conda create -n avt-detection python=3.10 -y
    conda create -n sar-mmdet3 python=3.10 -y
EOF

SHELL ["conda", "run", "-n", "sar-mmdet3", "/bin/bash", "-c"]

# TODO: fill the package version
RUN <<EOF
    apt install software-properties-common -y
    add-apt-repository ppa:ubuntu-toolchain-r/test -y
    apt-get install gcc-9 g++-9 -y
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9 
    update-alternatives --install /usr/bin/x86_64-linux-gnu-gcc x86_64-linux-gnu-gcc /usr/bin/gcc-9 60 --slave /usr/bin/x86_64-linux-gnu-g++ x86_64-linux-gnu-g++ /usr/bin/g++-9 

    mkdir -p /workspace/sar/
    cd /workspace/sar/

    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install -U openmim
    mim install mmengine==0.10.5
    mim install mmcv==2.1.0

#Install mmdetection
    rm -rf mmdetection
    git clone -bv3.3.0 https://github.com/open-mmlab/mmdetection.git
    cd mmdetection

    pip install -e .
EOF

SHELL ["conda", "run", "-n", "avt-detection", "/bin/bash", "-c"]

RUN <<EOF
    conda install pip=24.0 -y
    conda install conda-forge::ninja=1.12.1 -y
    pip install numpy==1.26.4
    pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
    pip install openmim

    mim install mmcv-full==1.5.3
    mim install mmdet==2.25.2

    # git clone --depth 1 --branch main  --single-branch https://github.com/open-mmlab/mmrotate.git 
    git clone --branch=v0.3.4 --single-branch https://github.com/open-mmlab/mmrotate.git 
    cd mmrotate 
    pip install -r requirements/build.txt
    pip install -v -e . 
    mim install mmrotate==0.3.4 
    pip install timm==0.6.13 future tensorboard

    cd  /workspace/avt-detection/
    pip install -r requirements/lsk.txt
    pip install -r requirements/app.txt
    pip install -r requirements/ship.txt
    cd ./LSKNet && pip install -v -e .
    pip cache purge
    conda clean --all -y
EOF
