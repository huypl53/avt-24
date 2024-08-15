# Steps to deploy AVT detection

## Clone source code

```bash
git clone --depth 1 --branch main --single-branch https://github.com/huypl53/avt-24 avt-detection
cd avt-detection
```

Download [epoch_3_050324.pth](https://www.google.com/url?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F1Rbys2P80YcovdYcJ1yrPcFs_OikHWAEG%2Fview%3Fusp%3Dsharing) to current directory then clone LSKNet

```bash
git clone --depth 1 --branch main --single-branch https://github.com/huypl53/LSKNet/ LSKNet
```

## Setup containers

```bash
# docker pull pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
docker compose -f docker/lsk/docker-compose.yml up
```
