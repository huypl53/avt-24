import os
import cv2
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dictdiffer import diff
from mmdet.apis import init_detector
from mmrotate.apis import inference_detector_by_patches
from mmseg.apis import inference_segmentor, init_segmentor
import logging

from app.schema import (
    DetectionInputParam,
    DetectionTaskType,
    EODetectionParam,
    ExtractedObject,
    ObjectCategory,
)
from app.service.binio import (
    ftpTransfer,
    read_ftp_bin_image,
    write_ftp_np_image,
    write_text_file,
)
from core.raster import RasterImage
from core.runway import Runway, process_runway_image
from core.ship.classifier import classify_ship
from utils.lsk import crop_rotated_rectangle, xywhr2xyxyxyxy
from utils.raster import (
    angle_to_bearings,
    latlon2meter,
    lonlat2meter,
    pixel_point_to_lat_long,
    read_tif_meta,
)
from logger import get_main_logger

logger = get_main_logger(
    __name__, log_file="./logs/image_processor.log", level=logging.INFO
)


class ImageProcessor:
    """Handles image processing operations including model management and inference."""

    def __init__(self):
        self.model = None
        self.model_runway = None
        self.reload_model = False
        self.tmp_im_path = ""
        self.im: Optional[np.ndarray] = None
        self.bname: str = ""
        self.save_dir: str = ""
        self.task_infer_image_success = False

    def clear_model(self, model):
        """Clear model from memory."""
        del model
        # import gc
        # gc.collect()
        # torch.cuda.empty_cache()

    def infer_image_runway(self, img) -> Optional[np.ndarray]:
        """Infer runway from image using segmentation model."""
        if self.model_runway is None:
            return None

        result = inference_segmentor(self.model_runway, img)
        runway_mask = result[0]
        if runway_mask is None:
            return None

        boundary_mask_img = np.where(runway_mask > 0, 255, 0).astype(np.uint8)
        cv2.imwrite(f"./tmp/{self.current_task_id}-runway-mask.png", boundary_mask_img)

        from utils.processing import mask2rbboxes

        runway_rbboxes = mask2rbboxes(boundary_mask_img)
        return runway_rbboxes

    def update_model_params(
        self, input_param_dict: Dict, pre_param_conf: EODetectionParam
    ) -> DetectionInputParam:
        """Update model parameters and reload if necessary."""
        input_param_no_file_dict = {
            k: v
            for k, v in input_param_dict.items()
            if k not in ["input_file", "checkpoint", "config"]
        }

        new_params = diff(
            input_param_no_file_dict,
            dict(pre_param_conf),
        )

        new_params_cnt = len(list(new_params))

        if new_params_cnt or not self.model:
            logger.info(f"new_params: {new_params}")
            if self.model:
                self.clear_model(self.model)
                self.model = None
            self.reload_model = True
            logger.info(f"new_params_cnt: {new_params_cnt}, task: {input_param_dict}")
            pre_param_conf = pre_param_conf.model_copy(update=input_param_no_file_dict)

        input_params = DetectionInputParam.model_validate(
            {
                **pre_param_conf.model_dump(),
                **input_param_no_file_dict,
                "input_file": input_param_dict["input_file"],
            }
        )

        if self.reload_model:
            self._load_models(input_params)

        return input_params

    def _load_models(self, input_params: DetectionInputParam):
        """Load detection and segmentation models."""
        try:
            torch.cuda.set_device(int(str(input_params.device).split(":")[-1]))
            self.model = init_detector(
                input_params.config,
                input_params.checkpoint,
                device=input_params.device,
            )
            logger.info(f"Loaded model: {input_params.config}")

            if self.model_runway is None:
                config_file = (
                    "/workspace/avt-detection/eo/runway_seg_config.py"
                    if not input_params.runway_config
                    else input_params.runway_config
                )
                checkpoint_file = (
                    "/workspace/avt-detection/eo/runway_seg_ckpt.pth"
                    if not input_params.runway_ckpt
                    else input_params.runway_ckpt
                )
                self.model_runway = init_segmentor(
                    config_file, checkpoint_file, device="cuda:0"
                )
                logger.info(f"Loaded runway model: {config_file}")

            self.reload_model = False
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.clear_model(self.model)
            self.model = None
            self.reload_model = True
            raise e

    async def process_image(
        self, input_file: str, task_id: int
    ) -> Tuple[Optional[np.ndarray], bool]:
        """Process image file and return image array and success status."""
        self.current_task_id = task_id
        self.bname = os.path.basename(input_file).rsplit(".", 1)[0]
        self.save_dir = os.path.join(self.input_params.out_dir, self.bname)
        ftpTransfer.mkdir(self.save_dir)

        try:
            bin_im = read_ftp_bin_image(input_file)
            self.task_infer_image_success = True
            if not bin_im:
                self.task_infer_image_success = False
                return None, False
        except Exception:
            self.task_infer_image_success = False
            return None, False

        self.tmp_im_path = f"./tmp/{self.bname}.tif"
        open(self.tmp_im_path, "wb").write(bin_im)

        image = np.asarray(bytearray(bin_im), dtype="uint8")
        self.im = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return self.im, True

    async def infer_image(self) -> Tuple[Optional[np.ndarray], bool]:
        """Perform inference on the loaded image."""
        try:
            result = inference_detector_by_patches(
                self.model,
                self.im,
                self.input_params.patch_sizes,
                self.input_params.patch_steps,
                self.input_params.img_ratios,
                self.input_params.merge_iou_thr,
            )
            self.task_infer_image_success = True
            return result, True
        except Exception as e:
            logger.error(e)
            self.task_infer_image_success = False
            return None, False

    def process_detection_results(
        self, classes_results: np.ndarray, image_id: str, im_th: int
    ) -> List[Dict]:
        """Process detection results and return extracted objects."""
        if not classes_results or not len(classes_results):
            return []

        image_detect_results: List[Dict] = []

        for class_id, class_rbboxes in enumerate(classes_results):
            output = np.array(class_rbboxes)
            output = output[output[..., -1] > self.input_params.score_thr]

            if not len(output):
                continue

            xyxyxyxy = xywhr2xyxyxyxy(output)
            output[..., 4] = np.degrees(output[..., 4])
            rbboxes = list(
                [
                    [
                        int(box[0]),
                        int(box[1]),
                        int(box[2]),
                        int(box[3]),
                        box[4],
                    ]
                    for box in output
                ]
            )
            valid_idx: List[int] = []
            skip = 0
            cls_names: List[str] = []

            for i, box in enumerate(rbboxes):
                patch = crop_rotated_rectangle(self.im, box)
                if patch is not None:
                    valid_idx.append(i)

                    lb_im_id = f"{class_id:03d}_{i-skip:04d}"
                    path = os.path.join(self.save_dir, lb_im_id)
                    patch_im_path = path + ".png"

                    write_ftp_np_image(patch, ".png", patch_im_path)
                    try:
                        if class_id == 1:
                            name = classify_ship(patch)
                        else:
                            name = ObjectCategory[class_id]
                    except:
                        name = str(DetectionTaskType.SHIP.value)
                    cls_names.append(name)
                else:
                    skip += 1

            output = output[valid_idx]
            xyxyxyxy = xyxyxyxy[valid_idx]
            flat_xy = xyxyxyxy.reshape(-1, 2)

            output = angle_to_bearings(output, 4)

            tif_meta = read_tif_meta(self.tmp_im_path)
            try:
                lat_long_center = pixel_point_to_lat_long(output[..., 0:2], tif_meta)
                latlong_xy = pixel_point_to_lat_long(flat_xy, tif_meta)
                latlong_xyxyxyxy = np.array(latlong_xy).reshape(-1, 4, 2)
                lat_long_wh = np.array(
                    [
                        [
                            lonlat2meter(
                                row[i][1],
                                row[i][0],
                                row[i + 1][1],
                                row[i + 1][0],
                            )
                            for i in range(2)
                        ]
                        for row in latlong_xyxyxyxy
                    ]
                )
                lat_long_wh = [wh if wh[0] < wh[1] else wh[::-1] for wh in lat_long_wh]
            except Exception:
                logger.error("Read crs from image failed!")
                continue

            lat_long_coords = np.concatenate(
                (lat_long_center, lat_long_wh, output[..., 4:]), axis=-1
            )

            for box_i, (cls_name, c) in enumerate(zip(cls_names, lat_long_coords)):
                lb_im_id = f"{class_id:03d}_{box_i:04d}"
                path = os.path.join(self.save_dir, lb_im_id)
                patch_lb_path = path + ".txt"
                patch_im_path = path + ".png"

                coords = c.tolist()
                write_text_file(" ".join([str(i) for i in coords]), patch_lb_path)

                detect_obj_id = f"{im_th:03d}-{lb_im_id}-{cls_name}"
                image_detect_results.append(
                    ExtractedObject(
                        id=detect_obj_id,
                        path=patch_im_path,
                        coords=coords,
                        lb_path=patch_lb_path,
                        class_id=cls_name,
                    ).model_dump()
                )

        return image_detect_results

    def process_runway_segmentation(self, image_id: str, im_th: int) -> List[Dict]:
        """Process runway segmentation and return runway objects."""
        raster_image = RasterImage(self.tmp_im_path)

        try:
            runways: List[Runway] = process_runway_image(
                self.im,
                self.infer_image_runway,
                pixel_to_latlon=lambda x, y: raster_image.pixel_to_coords(x, y),
                calculate_distance=lambda point1, point2: latlon2meter(
                    *point1, *point2
                ),
            )
        except Exception as e:
            logger.error(e)
            return []

        logger.info(
            f"Task id {self.current_task_id} has {len(runways)} runways. Details: {runways}"
        )

        runway_coords = [
            [
                [
                    r.center_point[0],
                    r.center_point[1],
                    r.width_meters,
                    r.length_meters,
                    r.angle,
                ]
                for r in runways
                if r.length_meters > self.input_params.runway_min_length or 500
            ]
        ]

        runway_results = [
            ExtractedObject(
                id=f"{im_th:03d}-{i:03d}-duong_bay",
                coords=coords,
                class_id="duong_bay",
            ).model_dump()
            for i, coords in enumerate(runway_coords)
        ]

        return runway_results

    def cleanup_temp_files(self):
        """Clean up temporary files."""
        if os.path.isfile(self.tmp_im_path):
            os.remove(self.tmp_im_path)

    def handle_memory_error(self):
        """Handle out of memory errors."""
        self.clear_model(self.model)
        self.model = None
        self.reload_model = True
        torch.cuda.synchronize()

    def set_input_params(self, input_params: DetectionInputParam):
        """Set input parameters for processing."""
        self.input_params = input_params

