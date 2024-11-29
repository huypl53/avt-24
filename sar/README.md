# SAR Ship Detection

## Training

> mmdetection 2.25.2

```bash
cd sar
git clone -q -b v2.25.2 https://github.com/open-mmlab/mmdetection.git
cd mmdetection
python tools/train.py /workspace/avt-detection/sar/sar_det_config.py
```
