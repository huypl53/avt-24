# Use an official PyTorch image with CUDA 11.3
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

# Set the working directory
WORKDIR /workspace

# Install OpenMIM
RUN pip install openmim

# Install required packages
RUN mim install mmcv-full==1.5.3 \
    && mim install mmdet==2.25.2

# Clone the mmrotate repository and install it
RUN git clone --depth=1 --single-branch main https://github.com/open-mmlab/mmrotate.git \
    && cd mmrotate \
    && pip install -r requirements/build.txt \
    && pip install -v -e . \
    && mim install mmrotate==0.3.4

# Install additional Python packages
RUN pip install timm==0.6.13 future tensorboard

COPY . /workspace/avt-detection
RUN echo 'Avt-deteciton directory.....' && ls
RUN echo 'Install app depenencies.....' && cd /workspace/avt-detection/ && pip install -f requirements.txt
RUN echo 'Install LSKnet depenencies.....' && cd /workspace/avt-detection/LSKNet && pip install -v -e .