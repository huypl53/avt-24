# Steps to deploy AVT detection

## Clone source code

```bash
git clone https://github.com/huypl53/avt-24 avt-detection
cd avt-detection
git clone https://github.com/huypl53/LSKNet/ LSKNet
```

## Setup containers

```bash
docker-compose -f docker/lsk/docker-compose.yml up
```
