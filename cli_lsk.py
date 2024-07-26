import asyncio
import json
import math
import os
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np
from mmdet.apis import init_detector
from mmrotate.apis import inference_detector_by_patches
from sqlalchemy import select

from app.db.connector import get_db
from app.model.task import TaskMd
from app.schema import DetectShipParam, ExtractedShip
from app.service.binio import ftpTransfer, read_ftp_bin_image, write_ftp_image
from log import logger
from utils.lsk import crop_rotated_rectangle, xywhr2xyxyxyxy
from utils.raster import haversine, pixel_point_to_lat_long


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
