# Image enhancement

## AVT containers

For building images

```bash
docker-compose -f ./docker/lsk/compose.yml up --build

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
