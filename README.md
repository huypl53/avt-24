# Image enhancement

> Python version 3.10.4

## Third-party tools

```bash
sudo apt-get install gdal-bin
```

## Brightness adjustment & CLAHE

```python
im = cv2.imread(im_path)
enhanced_im = adjust_gamma(im, 0.4)
enhanced_im = hist_equalize(im)
```

## Binary distribution

```bash
pyinstaller cli_enhance.py --onefile -n enhancing

cp ./dist/enhancing ~/bin/
```

## LSK inference

- First make sure that lsknet was clone into current directory by name 'lsknet'

```bash
git clone -q https://github.com/huypl53/LSKNet/ lsknet
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
