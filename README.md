# Image enhancement

## AVT containers

Build images and start containers for the first time:
> This only works when having internet

```bash
docker-compose -f ./docker/lsk/compose.yml up --build -d

# Then new container starts
# avt_ship_eo_detection: detect objects on EO images
# avt_ship_sar_detection: detect ships on SAR images
# avt_change_detection: detect chagnes on images
```

For save/load

```bash
docker save -o avt-lee.tar avt-lee:latest
docker load --input avt-lee.tar
```

Archive source code

```bash
# git archive --format=tar.gz -o avt-detection.tar.gz HEAD
tar --exclude-vcs -zcf avt-detection.tar.gz ./avt-detection/
```

Compress all into 1 file

```bash
tar -czf avt-AI.tar.gz avt-detection.tar.gz avt-lee.tar 
```

## Deployment

```bash
# 1. extract compressed file contained docker image and source code
tar -xzf avt-AI.tar.gz

# 2. load docker image
docker load --input avt-lee.tar

# 3. extract source code
tar -xzf avt-detection.tar.gz

# 4. cd to source code directory at './avt-detection/' and start all containers
cd avt-detection
docker compose -f ./docker/lsk/compose.yml up -d
```
