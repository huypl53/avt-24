import asyncio
import json
import math
import os
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from dictdiffer import diff
from mmdet.apis import init_detector
from mmrotate.apis import inference_detector_by_patches
from sqlalchemy import select

from app.db.connector import get_db
from app.model.task import TaskMd
from app.schema import DetectShipParam, ExtractedShip
from app.service.binio import ftpTransfer, read_ftp_bin_image, write_ftp_image
from log import logger
from utils.lsk import crop_rotated_rectangle, xywhr2xyxyxyxy
from utils.raster import latlong2meter, pixel_point_to_lat_long


async def async_main():
    # assert len(sys.argv) < 2
    # task_id = int(sys.argv[1])

    params: DetectShipParam = {}
    model = None
    current_task = None
    reload_model = False

    # counter = 0
    while True:
        # counter += 1
        a_session = anext(get_db())
        session = await a_session
        stmt = (
            select(TaskMd)
            # .where(TaskMd.id == task_id)
            .where(TaskMd.task_type == 5)  # task type of ship detection
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
                t.task_stat = 2  # task is checked
                session.commit()

                # if i == 1:
                #     break  # update only one
                param_dict = json.loads(t.task_param)
                new_params_cnt = len(list(diff(param_dict, dict(params))))
                if new_params_cnt:
                    reload_model = True
                    default_conf = DetectShipParam().model_dump()
                    default_conf.update(param_dict)
                    params = DetectShipParam(**default_conf)

                if reload_model:
                    model = init_detector(
                        params.config, params.checkpoint, device=params.device
                    )
                    reload_model = False

                bin_im = read_ftp_bin_image(params.input_file)
                if not bin_im:
                    t.task_stat = 0
                    t.task_message = f"Read image failed at {params.input_file}"
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

                # TODO: handle score thresh
                output = output[output[..., -1] > params.score_thr]

                if not len(output):
                    t.task_stat = 1
                    t.task_message = "No above-score detection"
                xyxyxyxy = xywhr2xyxyxyxy(output)
                output[..., 4] = np.degrees(output[..., 4])
                rbboxes = [
                    [(int(r[0]), int(r[1])), (int(r[2]), int(r[3])), r[4]]
                    for r in output
                ]

                # Pick only valid boxes that provide rectangle
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
                flat_xy = xyxyxyxy.reshape(-1, 2)

                bname = os.path.basename(params.input_file).rsplit(".", 1)[0]
                tmp_im_path = f"./tmp/{bname}.tif"
                open(tmp_im_path, "wb").write(bin_im)
                save_dir = os.path.join(params.out_dir, bname)
                ftpTransfer.mkdir(save_dir)

                # Convert angles to `Bearings maths`
                output[..., 4] -= 90
                angles = output[..., 4]
                output[..., 4][angles > 0] = 360 - angles[angles > 0]
                output[..., 4][angles < 0] = -angles[angles < 0]

                try:
                    lat_long_center = pixel_point_to_lat_long(
                        tmp_im_path, output[..., 0:2]
                    )
                except:
                    t.task_stat = 0
                    t.task_message = "Read coordinates from image failed!"
                    continue

                latlong_xy = pixel_point_to_lat_long(tmp_im_path, flat_xy)
                latlong_xyxyxyxy = np.array(latlong_xy).reshape(-1, 4, 2)
                # lat_long_wh = pixel_point_to_lat_long(tmp_im_path, output[..., 2:4])
                lat_long_wh = np.array(
                    [
                        [
                            latlong2meter(
                                row[i][1], row[i][0], row[i + 1][1], row[i + 1][0]
                            )
                            for i in range(2)
                        ]
                        for row in latlong_xyxyxyxy
                    ]
                )

                # [[lat_c, lon_c, w_meter, h_meter, score, angle_degree],]
                lat_long_coords = np.concatenate(
                    (lat_long_center, lat_long_wh, output[..., 4:]), axis=-1
                )

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
                if os.path.isfile(tmp_im_path):
                    os.remove(tmp_im_path)
        except Exception as e:
            logger.error(e)
            if current_task:
                current_task.task_message = str(e)
        finally:
            await session.commit()

        print("----------")
        await asyncio.sleep(3)
        # await session.close()


# Run this from outter directory
# python ./lsknet/huge_images_extract.py --dir ./images  --config './lsknet/configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90.py' --checkpoint './epoch_3_050324.pth' --score-thr 0.5 --save-dir /tmp/ships/

if __name__ == "__main__":
    # cli_main()
    asyncio.run(async_main())
