# Image enhancement

> Python version 3.10.4

## Env config

To update log level, database and ftp, please modify relevant values in .env

## Third-party tools

```bash
sudo apt-get install gdal-bin
```

## Binary distribution

```bash
pyinstaller cli_enhance.py --onefile -n enhancing

cp ./dist/enhancing ~/bin/
```

## LSK inference

- First make sure that LSKNet was clone into current directory by name 'LSKNet'

```bash
git clone -q https://github.com/huypl53/LSKNet/ LSKNet
```

- Install dependencies

```bash
bash ./scripts/install_requirements.sh
```

- Start program

```bash
bash -i <path/to/scripts/run_lsk.sh>
```

## Anomaly detections

### Reed-xiaoli

```bash
# find the anomaly area
# save the mask.png and export anomaly areas to .txt, each line consists of keypoints
python ./anomaly/rx.py <path/to/image>

# draw the anomaly albel
python ./anomaly/rx_draw.py <path/to/image> <path/to/label.txt>
```
