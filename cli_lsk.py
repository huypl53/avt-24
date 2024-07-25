import asyncio
import json
import math
import os
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from mmdet.apis import init_detector
from mmrotate.apis import inference_detector_by_patches
from sqlalchemy import select

from app.db.connector import get_db
from app.model.task import TaskMd
from app.schema import DetectShipParam, ExtractedShip
from app.service.binio import ftpTransfer, read_ftp_bin_image, write_ftp_image
from log import logger
from utils.raster import haversine, pixel_point_to_lat_long


def image_rotate_without_crop(mat, angle):
    # https://stackoverflow.com/questions/22041699/rotate-an-image-without-cropping-in-opencv-in-c
    # angle in degrees

    height, width = mat.shape[:2]
    image_center = (width / 2, height / 2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)

    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))

    return rotated_mat


def crop_rectangle(image, rect):
    # rect has to be upright

    num_rows = image.shape[0]
    num_cols = image.shape[1]

    if not inside_rect(rect=rect, num_cols=num_cols, num_rows=num_rows):
        print("Proposed rectangle is not fully in the image.")
        return None

    rect_center = rect[0]
    rect_center_x = rect_center[0]
    rect_center_y = rect_center[1]
    rect_width = rect[1][0]
    rect_height = rect[1][1]

    return image[
        rect_center_y
        - rect_height // 2 : rect_center_y
        + rect_height
        - rect_height // 2,
        rect_center_x - rect_width // 2 : rect_center_x + rect_width - rect_width // 2,
    ]


def rect_bbx(rect):
    # Rectangle bounding box for rotated rectangle
    # Example:
    # rotated rectangle: height 4, width 4, center (10, 10), angle 45 degree
    # bounding box for this rotated rectangle, height 4*sqrt(2), width 4*sqrt(2), center (10, 10), angle 0 degree

    box = cv2.boxPoints(rect)

    x_max = int(np.max(box[:, 0]))
    x_min = int(np.min(box[:, 0]))
    y_max = int(np.max(box[:, 1]))
    y_min = int(np.min(box[:, 1]))

    center = (int((x_min + x_max) // 2), int((y_min + y_max) // 2))
    width = int(x_max - x_min)
    height = int(y_max - y_min)
    angle = 0

    return (center, (width, height), angle)


def inside_rect(rect, num_cols, num_rows):
    # Determine if the four corners of the rectangle are inside the rectangle with width and height
    # rect tuple
    # center (x,y), (width, height), angle of rotation (to the row)
    # center  The rectangle mass center.
    # center tuple (x, y): x is regarding to the width (number of columns) of the image, y is regarding to the height (number of rows) of the image.
    # size    Width and height of the rectangle.
    # angle   The rotation angle in a clockwise direction. When the angle is 0, 90, 180, 270 etc., the rectangle becomes an up-right rectangle.
    # Return:
    # True: if the rotated sub rectangle is side the up-right rectange
    # False: else

    rect_center = rect[0]
    rect_center_x = rect_center[0]
    rect_center_y = rect_center[1]

    rect_width, rect_height = rect[1]

    rect_angle = rect[2]

    if (rect_center_x < 0) or (rect_center_x > num_cols):
        return False
    if (rect_center_y < 0) or (rect_center_y > num_rows):
        return False

    # https://docs.opencv.org/3.0-beta/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html
    box = cv2.boxPoints(rect)

    x_max = int(np.max(box[:, 0]))
    x_min = int(np.min(box[:, 0]))
    y_max = int(np.max(box[:, 1]))
    y_min = int(np.min(box[:, 1]))

    if (x_max <= num_cols) and (x_min >= 0) and (y_max <= num_rows) and (y_min >= 0):
        return True
    else:
        return False


def crop_rotated_rectangle(image, rect):
    # Crop a rotated rectangle from a image

    num_rows = image.shape[0]
    num_cols = image.shape[1]

    if not inside_rect(rect=rect, num_cols=num_cols, num_rows=num_rows):
        print("Proposed rectangle is not fully in the image.")
        return None

    rotated_angle = rect[2]

    rect_bbx_upright = rect_bbx(rect=rect)
    rect_bbx_upright_image = crop_rectangle(image=image, rect=rect_bbx_upright)

    rotated_rect_bbx_upright_image = image_rotate_without_crop(
        mat=rect_bbx_upright_image, angle=rotated_angle
    )

    rect_width = rect[1][0]
    rect_height = rect[1][1]

    crop_center = (
        rotated_rect_bbx_upright_image.shape[1] // 2,
        rotated_rect_bbx_upright_image.shape[0] // 2,
    )

    return rotated_rect_bbx_upright_image[
        crop_center[1]
        - rect_height // 2 : crop_center[1]
        + (rect_height - rect_height // 2),
        crop_center[0]
        - rect_width // 2 : crop_center[0]
        + (rect_width - rect_width // 2),
    ]


def xywhr2xyxyxyxy(x):
    """
    Convert batched Oriented Bounding Boxes (OBB) from [xywh, rotation] to [xy1, xy2, xy3, xy4]. Rotation values should
    be in degrees from 0 to 90.

    Args:
        x (numpy.ndarray | torch.Tensor): Boxes in [cx, cy, w, h, rotation] format of shape (n, 5) or (b, n, 5).

    Returns:
        (numpy.ndarray | torch.Tensor): Converted corner points of shape (n, 4, 2) or (b, n, 4, 2).
    """
    cos, sin, cat, stack = (
        (torch.cos, torch.sin, torch.cat, torch.stack)
        if isinstance(x, torch.Tensor)
        else (np.cos, np.sin, np.concatenate, np.stack)
    )

    ctr = x[..., :2]
    w, h, angle = (x[..., i : i + 1] for i in range(2, 5))
    cos_value, sin_value = cos(angle), sin(angle)
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1 = cat(vec1, -1)
    vec2 = cat(vec2, -1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return stack([pt1, pt2, pt3, pt4], -2)


async def async_main():
    # assert len(sys.argv) < 2
    # task_id = int(sys.argv[1])
    a_session = anext(get_db())
    session = await a_session

    params: DetectShipParam = None
    model = None
    current_task = None
    reload_model = False

    # counter = 0
    while True:
        # counter += 1
        stmt = (
            select(TaskMd)
            # .where(TaskMd.id == task_id)
            .where(TaskMd.type == 5)  # task type of ship detection
            .where(TaskMd.task_stat < 0)
            .order_by(TaskMd.task_stat.desc())
        )
        results = await session.execute(stmt)
        mapping_results = results.mappings().all()
        tasks: List[TaskMd] = [m["TaskMd"] for m in mapping_results]

        print("----------")
        try:
            for i, t in enumerate(tasks):
                current_task = t
                if i == 1:
                    break  # update only one
                param_dict = json.loads(t.task_param)
                if not t.task_param and "input_file" in param_dict:
                    params = DetectShipParam(input_file=param_dict["input_file"])
                    reload_model = True
                elif (not params) or t.task_param != params.model_dump():
                    try:
                        params = DetectShipParam(**param_dict)
                        reload_model = True
                    except Exception as e:
                        reload_model = False
                        t.task_stat = 0
                        t.task_message = str(e)
                else:
                    t.task_stat = 0  # task got error
                    t.task_message = "Init model failed!"
                    reload_model = False
                    continue
                if reload_model:
                    model = init_detector(
                        params.config, params.checkpoint, device=params.device
                    )

                bin_im = read_ftp_bin_image(params.input_file)
                image = np.asarray(bytearray(bin_im), dtype="uint8")
                im = cv2.imdecode(image, cv2.IMREAD_COLOR)

                result = inference_detector_by_patches(
                    model,
                    im,
                    params.patch_sizes,
                    params.patch_steps,
                    params.img_ratios,
                    params.merge_iou_thr,
                )  # inference for batch
                if not len(result):
                    t.task_stat = 1
                    t.task_message = "No detection"
                    continue
                output = np.array(result[0])  # take first list of boxes from batch
                output = output[output[..., -1] > params.score_thr]
                xyxyxyxy = xywhr2xyxyxyxy(output)
                output[..., 4] = np.degrees(output[..., 4])
                rbboxes = [
                    [(int(r[0]), int(r[1])), (int(r[2]), int(r[3])), r[4]]
                    for r in output
                ]

                valid_idx: List[int] = []
                patches: List[np.ndarray] = []
                for i, box in enumerate(rbboxes):
                    patch = crop_rotated_rectangle(
                        im, box
                    )  # patch if None if crop failed
                    if patch is not None:
                        patches.append(patch)
                        valid_idx.append(i)

                output = output[valid_idx]
                xyxyxyxy = xyxyxyxy[valid_idx]
                # flat_xyxyxyxy = xyxyxyxy.reshape(-1, 2)

                lat_long_center = pixel_point_to_lat_long(tmp_im_path, output[..., 0:2])
                # lat_long_wh = pixel_point_to_lat_long(tmp_im_path, output[..., 2:4])
                lat_long_wh = np.array(
                    [
                        haversine(row[i][1], row[i][0], row[i + 1][1], row[i + 1][0])
                        for i in range(2)
                    ]
                    for row in xyxyxyxy
                )

                # [[lat_c, lon_c, w_meter, h_meter, score, angle_degree],]
                lat_long_coords = np.concatenate(
                    (lat_long_center, lat_long_wh, output[..., 4:])
                )

                bname = os.path.basename(params.input_file).rsplit(".", 1)[0]
                tmp_im_path = f"/tmp/{bname}.tif"
                open(tmp_im_path, "wb").write(bin_im)
                save_dir = os.path.join(params.out_dir, bname)
                ftpTransfer.mkdir(save_dir)

                detect_results: List[Dict] = []
                for i, (p, c) in enumerate(zip(patches, lat_long_coords)):
                    im_id = f"{i:03d}"
                    path = os.path.join(save_dir, im_id) + ".png"
                    # Box xyxyxyxy
                    # coords = xy.tolist()

                    # Box cx, cy, w, h, angle
                    coords = c.tolist()
                    write_ftp_image(p, ".png", path)
                    detect_results.append(
                        ExtractedShip(id=im_id, path=path, coords=coords).model_dump()
                    )
                # TODO: xyxyxyxy to real coordinates
                # t.task_output = json.dumps(xyxyxyxy.tolist())
                t.task_output = json.dumps(detect_results)
                t.task_stat = 1
                t.task_message = "Successfully"
                t.task_param = json.dumps(params.model_dump())
                t.process_id = os.getpid()
        except Exception as e:
            logger.error(e)
            if current_task:
                current_task.task_message = str(e)

        print("----------")
        await session.commit()
        await asyncio.sleep(3)
        # await session.close()


# Run this from outter directory
# python ./lsknet/huge_images_extract.py --dir ./images  --config './lsknet/configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90.py' --checkpoint './epoch_3_050324.pth' --score-thr 0.5 --save-dir /tmp/ships/

if __name__ == "__main__":
    # cli_main()
    asyncio.run(async_main())
