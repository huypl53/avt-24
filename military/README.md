# Config generation for mmdet and mmseg

Copy config files into mmdet, mmseg correspondingly then run them

## Runway setup

### mmsegment v0.30.0

Annotaion image pixel value should fall in range `[0, num_classes - 1]` [link](https://github.com/open-mmlab/mmsegmentation/blob/v0.30.0/docs/en/tutorials/customize_datasets.md)

Custom config should follow [tutorial](https://github.com/open-mmlab/mmsegmentation/blob/v0.30.0/docs/en/tutorials/config.md)

[Advanced tips](https://github.com/open-mmlab/mmsegmentation/blob/v0.30.0/docs/en/tutorials/training_tricks.md) can be considered later

[DeepLabV3 (FP16)](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/deeplabv3/README.md) is used

```bash
cp seg_runway_config.py /workspace/mmsegmentation
cd  /workspace/mmsegmentation
# download weighs here
wget https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r101-d8_fp16_512x1024_80k_cityscapes/deeplabv3_r101-d8_fp16_512x1024_80k_cityscapes_20200717_230920-774d9cec.pth

python seg_runway_config.py # this generates runway_config.py
# mv runway_config.py configs/deeplabv3/

cp runway_dataset.py /workspace/mmsegmentation/mmseg/datasets/runway_dataset.py
# update mmsegmentation/mmseg/datasets/__init__.py to import from  runway_dataset.py

python tools/train.py configs/deeplabv3/runway_config.py --load-from=deeplabv3_r101-d8_fp16_512x1024_80k_cityscapes_20200717_230920-774d9cec.pth
```

### mmsegment v1.2.2

> DeepLabV3 R-18-D8 769x769 80000 1.9 5.55 V100 76.60 78.26

```bash
cp seg_runway_config.py /workspace/mmseg122
cd  /workspace/mmseg122

wget https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r18-d8_769x769_80k_cityscapes/deeplabv3_r18-d8_769x769_80k_cityscapes_20201225_021506-6452126a.pth

python seg_runway_config.py # this generates runway_config.py
# mv runway_config.py configs/deeplabv3/

cp runway_dataset.py /workspace/mmseg122/mmseg/datasets/runway_dataset.py
# update mmsegmentation/mmseg/datasets/__init__.py to import from  runway_dataset.py

python tools/train.py configs/deeplabv3/runway_config.py --load-from=deeplabv3_r18-d8_769x769_80k_cityscapes_20201225_021506-6452126a.pth

```

### Post-process

> If there is still a need for post-processing, other techniques such as Hough transform, nearest neighbor clustering for sparsely distributed points like DBSCAN, and OPTICS algorithms can refine the predictions further before applying any thresholds. [link](https://mosaicatm.com/2019/09/09/detecting-airport-layouts-aviation-computer-vision/)
