# Image enhancement

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
```bash
bash -i <path/to/scripts/run_lsk.sh>
```
