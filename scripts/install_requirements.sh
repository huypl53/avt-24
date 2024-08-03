#!/bin/bash

cd './lsknet/' # previously clone at https://github.com/huypl53/LSKNet/ 
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
# pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
pip install openmim

# notice version: https://github.com/open-mmlab/mmrotate/blob/main/docs/en/faq.md#mmcv-installation
mim install mmcv-full==1.5.3
mim install mmdet==2.25.2

# mmrotate 0.3.4
git clone https://github.com/open-mmlab/mmrotate.git
cd mmrotate
pip install -r requirements/build.txt
pip install -v -e .
mim install mmrotate==0.3.4

pip install timm==0.6.13
pip install future tensorboard

pip install numpy==1.21.5 
pip install opencv_python_headless==4.6.0.66 
pip install pydantic==2.7.4
pip install pydantic_core==2.18.4
pip install pydantic_settings==2.3.4
pip install SQLAlchemy==2.0.31
pip install geopy
pip install dictdiffer==0.9.0
pip install matplotlib==3.9.1