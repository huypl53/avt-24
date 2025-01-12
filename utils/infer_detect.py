import mmcv
import numpy as np
from mmdet.apis import inference_detector, init_detector
from mmengine.config import Config

try:
    from sahi.slicing import slice_image
except ImportError:
    raise ImportError(
        'Please run "pip install -U sahi" '
        "to install sahi first for large image inference."
    )

from mmdet.utils.large_image import merge_results_by_nms


def inference_detector_by_patches(
    model,
    img: np.ndarray,
    patch_size: int = 640,
    patch_overlap_ratio: float = 0.25,
    merge_iou_thr: float = 0.25,
    merge_nms_type: str = "nms",
    batch_size: int = 1,
):
    # arrange slices
    height, width = img.shape[:2]
    sliced_image_object = slice_image(
        img,
        slice_height=patch_size,
        slice_width=patch_size,
        auto_slice_resolution=False,
        overlap_height_ratio=patch_overlap_ratio,
        overlap_width_ratio=patch_overlap_ratio,
    )
    # perform sliced inference
    slice_results = []
    start = 0
    while True:
        # prepare batch slices
        end = min(start + batch_size, len(sliced_image_object))
        images = []
        for sliced_image in sliced_image_object.images[start:end]:
            images.append(sliced_image)

        # forward the model
        slice_results.extend(inference_detector(model, images))

        if end >= len(sliced_image_object):
            break
        start += batch_size

    img = mmcv.imconvert(img, "bgr", "rgb")

    image_result = merge_results_by_nms(
        slice_results,
        sliced_image_object.starting_pixels,
        src_image_shape=(height, width),
        nms_cfg={"type": merge_nms_type, "iou_threshold": merge_iou_thr},
    )

    return image_result
