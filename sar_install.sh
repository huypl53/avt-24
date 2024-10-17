#!/bin/bash
mkdir -p /workspace/sar/
cd /workspace/sar/

conda create -n mmdet3 python==3.10
conda activate mmdet3
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -U openmim
mim install mmengine==0.10.5
mim install mmcv==2.2.0

#Install mmdetection
rm -rf mmdetection
git clone -bv3.1.0 https://github.com/open-mmlab/mmdetection.git
cd mmdetection

pip install -e .