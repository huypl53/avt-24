# Steps to deploy AVT detection

## Clone source code

```bash
git clone --depth 1 --branch main --single-branch https://github.com/huypl53/avt-24 avt-detection
cd avt-detection
```

Download [epoch_3_050324.pth](https://www.google.com/url?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F1Rbys2P80YcovdYcJ1yrPcFs_OikHWAEG%2Fview%3Fusp%3Dsharing) to current directory

Download [redet_re50_fpn_1x_dota_ms_rr_le90-fc9217b5.pth](https://drive.google.com/file/d/1P36WSrFXynaOOIDVvIDrW2jZy-_pCy8C/view?usp=sharing) and [roi_trans_r50_fpn_1x_dota_ms_rr_le90-fa99496f.pth](https://drive.google.com/file/d/15yzkFTf2Mdh0P_McjiUnWhAaQOL-wKW6/view?usp=drive_link)

Then clone LSKNet

```bash
git clone --depth 1 --branch main --single-branch https://github.com/huypl53/LSKNet/ LSKNet
```

## Setup containers

Update `.env` file

First, build the desired docker image

```base
 docker compose -f docker/lsk/base.yml build lsknet
```

The, start new containers

```bash
# docker pull pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
docker compose -f docker/lsk/base.yml --profile ship up
docker compose -f docker/lsk/base.yml --profile change up
docker compose -f docker/lsk/base.yml --profile military up
```

### Militia object detection demo

cd to `LSKNet` then run one of:

```bash
python demo/huge_image_demo.py demo/dota_demo.jpg configs/redet/redet_re50_refpn_1x_dota_ms_rr_le90.py ../redet_re50_fpn_1x_dota_ms_rr_le90-fc9217b5.pth

python demo/huge_image_demo.py demo/dota_demo.jpg configs/roi_trans/roi_trans_r50_fpn_1x_dota_ms_rr_le90.py ../roi_trans_r50_fpn_1x_dota_ms_rr_le90-fa99496f.pth
```

> Label order: plane, ship, storage tank, baseball diamond, tennis court, basketball court, ground track field, harbor, bridge, large vehicle, small vehicle, helicopter, roundabout, soccer ball field and swimming pool. Model prediction format has shape class_num x [5 for bbox | 6 for oriented bbox]
