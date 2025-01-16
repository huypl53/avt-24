import asyncio
import json
import multiprocessing
import multiprocessing.synchronize
import os
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
from dictdiffer import diff
from mmdet.apis import init_detector
from mmrotate.apis import inference_detector_by_patches
from mmseg.apis import inference_segmentor, init_segmentor
from sqlalchemy import select, text
from sqlalchemy.exc import InterfaceError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession
import argparse

from app.db.connector import get_db
from app.model.task import TaskMd
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
from core.box_record import BoxDetect
from core.raster import RasterImage
from core.runway import Runway, process_runway_image
from core.segment_slice import SlidingWindowInference
from core.ship.classifier import classify_ship
from log import logger
from utils.lsk import crop_rotated_rectangle, xywhr2xyxyxyxy
from utils.processing import (
    find_boundary_keypoints,
    get_rotated_bbox_corners,
    mask2rbboxes,
)
from utils.raster import (
    angle_to_bearings,
    latlon2meter,
    lonlat2meter,
    pixel_point_to_lat_long,
    read_tif_meta,
)


async def async_main():
    while True:
        task_type = DetectionTaskType.SHIP
        _i += 1

        stmt_task = (
            select(TaskMd)
            # .where(TaskMd.id == task_id)
            .where(TaskMd.task_type == task_type.value)  # task type of ship detection
            .where(TaskMd.task_stat < 0)
            .order_by(TaskMd.task_stat.desc())
        )
        tasks = await query_tasks_by_stmt(stmt_task, session)

        try:
            for task_i, t in enumerate(tasks):
                current_task = t
                extra_mesg = ""
                if t.task_id_ref and t.task_id_ref != 0:
                    # t has to wait to task with id = t.task_id_ref
                    stmt_ref_tasks = (
                        select(TaskMd)
                        .where(
                            TaskMd.id == t.task_id_ref
                        )  # task type of ship detection
                        .where(TaskMd.task_stat == 1)
                        .order_by(TaskMd.task_stat.desc())
                    )
                    sub_tasks = await query_tasks_by_stmt(stmt_ref_tasks, session)
                    if len(sub_tasks) == 0:
                        msg = "Waiting for task id = {}".format(t.task_id_ref)
                        await _update_task(msg)
                        continue
                msg = "Task is being processed"
                t.process_id = os.getpid()
                await _update_task(msg)

                input_param_dict = parse_param_dict(t.task_param)
                if "image_type" not in input_param_dict:
                    await _update_task("<image_type> field is requried!", 0)
                    continue
                if input_param_dict["image_type"] != "EO":
                    continue
                if "input_file" not in input_param_dict:
                    await _update_task("<input_file> field is requried!", 0)
                    continue

                _update_process_func(t)
                _update_param(input_param_dict)

                # t.task_param = stringify_dict_list(input_params.model_dump())
                t.task_param = input_params.model_dump_json(exclude_none=True)
                await _update_task()
                detect_results = []
                detection_history: List[List[List[BoxDetect]]] = [
                    [] for _ in range(len(input_params.input_file))
                ]
                seg_runway_results = []
                for im_th, image_path in enumerate(input_params.input_file):
                    image_id = image_path
                    _, success = await _process_image(image_path)
                    if not success:
                        continue
                    classes_results, success = await _infer_image_params()
                    # if not success:
                    #     continue
                    if success and classes_results is not None and len(classes_results):
                        # await _update_task("No detection", 1)

                        # TODO: handle score thresh
                        # classes_results = np.array(classes_results)
                        image_detect_results: List[Dict] = []
                        for class_id, class_rbboxes in enumerate(classes_results):
                            detection_history[im_th].append([])
                            # output = result[:, result[..., -1] > input_params.score_thr]
                            output = np.array(
                                class_rbboxes
                            )  # [[cx, cy, w, h, angle, score]] all in pixel, angle in radian
                            output = output[output[..., -1] > input_params.score_thr]

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
                            # patches: List[np.ndarray] = []

                            skip = 0
                            cls_names: List[str] = []
                            for i, box in enumerate(rbboxes):
                                patch = crop_rotated_rectangle(
                                    im, box
                                )  # patch if None if crop failed
                                if patch is not None:
                                    # patches.append(patch)
                                    valid_idx.append(i)

                                    lb_im_id = f"{class_id:03d}_{i-skip:04d}"
                                    path = os.path.join(save_dir, lb_im_id)
                                    # patch_lb_path = path + ".txt"
                                    patch_im_path = path + ".png"

                                    write_ftp_np_image(patch, ".png", patch_im_path)
                                    try:
                                        if class_id == 1:
                                            name = classify_ship(patch)
                                        else:
                                            name = ObjectCategory[class_id]
                                    except:
                                        extra_mesg += "Classify ship failed!"
                                        name = str(DetectionTaskType.SHIP.value)
                                    cls_names.append(name)
                                else:
                                    skip += 1

                            output = output[valid_idx]
                            xyxyxyxy = xyxyxyxy[valid_idx]
                            flat_xy = xyxyxyxy.reshape(-1, 2)

                            output = angle_to_bearings(output, 4)

                            tif_meta = read_tif_meta(tmp_im_path)
                            try:
                                lat_long_center = pixel_point_to_lat_long(
                                    output[..., 0:2], tif_meta
                                )
                                latlong_xy = pixel_point_to_lat_long(flat_xy, tif_meta)
                                latlong_xyxyxyxy = np.array(latlong_xy).reshape(
                                    -1, 4, 2
                                )
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
                                lat_long_wh = [
                                    wh if wh[0] < wh[1] else wh[::-1]
                                    for wh in lat_long_wh
                                ]
                            except Exception:
                                await _update_task("Read crs from image failed!")
                                continue
                            lat_long_coords = np.concatenate(
                                (lat_long_center, lat_long_wh, output[..., 4:]), axis=-1
                            )
                            if task_type == DetectionTaskType.SHIP:
                                # match_adsb_indices = await check_adsb(lat_long_coords)
                                # if match_adsb_indices is not None:
                                #     patches = [
                                #         p
                                #         for i, p in enumerate(patches)
                                #         if i in match_adsb_indices
                                #     ]
                                #     lat_long_coords = lat_long_coords[match_adsb_indices]
                                pass
                            for box_i, (cls_name, c) in enumerate(
                                zip(cls_names, lat_long_coords)
                            ):
                                lb_im_id = f"{class_id:03d}_{box_i:04d}"
                                path = os.path.join(save_dir, lb_im_id)
                                patch_lb_path = path + ".txt"
                                patch_im_path = path + ".png"
                                # Box cx, cy, w, h, angle
                                coords = c.tolist()
                                write_text_file(
                                    " ".join([str(i) for i in coords]), patch_lb_path
                                )

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

                        detect_results.append(
                            {"image_id": image_id, "detections": image_detect_results}
                        )
                    # -----Segment runway--------
                    raster_image = RasterImage(tmp_im_path)

                    # TODO: no need to replace image data
                    # raster_image.replace_image_data(im)

                    # slicer = SlidingWindowInference(
                    #     inference_fn=infer_image_runway,
                    #     window_size=(1024, 1024),
                    #     smoothier=True,
                    # )

                    # runway_mask = slicer(im)
                    # runway_rbboxes = infer_image_runway(im)
                    # if runway_rbboxes is None:
                    #     continue

                    try:
                        runways: List[Runway] = process_runway_image(
                            im,
                            infer_image_runway,
                            pixel_to_latlon=lambda x, y: raster_image.pixel_to_coords(
                                x, y
                            ),
                            calculate_distance=lambda point1, point2: latlon2meter(
                                *point1, *point2
                            ),
                        )

                    except Exception as e:
                        logger.error(e)
                        continue

                    # runway_rbboxes = [
                    #     bbox for bbox in runway_rbboxes if max(bbox[1]) > 300
                    # ]
                    logger.info(
                        f"Task id {t.id} has {len(runways)} runways. Details: {runways}"
                    )

                    # runway_xyxyxyxy = [
                    #     get_rotated_bbox_corners(rbbox) for rbbox in runway_rbboxes
                    # ]

                    # runway_xy = np.array(runway_xyxyxyxy).reshape(-1, 2)
                    # runway_lat_lon_xy = [
                    #     raster_image.pixel_to_coords(xy[0], xy[1]) for xy in runway_xy
                    # ]
                    # runway_lat_lon_xyxyxyxy = np.array(runway_lat_lon_xy).reshape(-1, 8)

                    # runway_lat_lon_wh = np.array(
                    #     [
                    #         [
                    #             latlong2meter(
                    #                 row[i + 1],
                    #                 row[i],
                    #                 row[i + 3],
                    #                 row[i + 2],
                    #             )
                    #             for i in range(0, 3, 2)
                    #         ]
                    #         for row in runway_lat_lon_xyxyxyxy
                    #     ]
                    # )

                    # runway_lat_lon_wh = [
                    #     wh if wh[0] < wh[1] else wh[::-1] for wh in runway_lat_lon_wh
                    # # ]

                    # runway_center_lat_lon = [
                    #     raster_image.pixel_to_coords(*rbbox[0])
                    #     for rbbox in runway_rbboxes
                    # ]
                    # runway_coords = np.array(
                    #     [
                    #         [center[0], center[1], wh[0], wh[1], rbbox[-1]]
                    #         for center, wh, rbbox in zip(
                    #             runway_center_lat_lon, runway_lat_lon_wh, runway_rbboxes
                    #         )
                    #     ]
                    # )
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
                            if r.length_meters > input_params.runway_min_length or 500
                        ]
                    ]

                    seg_runway_results.append(
                        {
                            "image_id": image_id,
                            "runway": [
                                ExtractedObject(
                                    id=f"{im_th:03d}-{i:03d}-duong_bay",
                                    coords=coords,
                                    class_id="duong_bay",
                                ).model_dump()
                                for i, coords in enumerate(runway_coords)
                            ],
                        }
                    )

                output_dict = [
                    image_result["detections"] for image_result in detect_results
                ]
                output_dict += [
                    image_result["runway"] for image_result in seg_runway_results
                ]
                if not task_infer_image_success:
                    await _update_task("Task inference failed!", 0)

                t.task_output = json.dumps(output_dict)
                t.task_stat = 1
                t.task_message = "\n".join(["Successfully", extra_mesg])
                if os.path.isfile(tmp_im_path):
                    os.remove(tmp_im_path)
                logger.info(f"Process task id = {t.id} successfully")

                stop_update_task_continuously()
                await asyncio.sleep(2)
                await _update_task(stat=1)
        except RuntimeError as e:
            stop_update_task_continuously()
            if "out of memory" not in str(e):
                pass
            else:
                clear_model(model)
                model = None
                reload_model = True
                torch.cuda.synchronize()
            logger.error(str(e))
            if current_task:
                await asyncio.sleep(2)
                await _update_task(str(e), 0)
            await asyncio.sleep(60)
        except (InterfaceError, OperationalError) as e:
            stop_update_task_continuously()
            logger.error(f"Connection error occurred: {e}")
            if session.is_active:
                await session.close()  # Close invalid session
            a_session = anext(get_db("main_task"))
            session = await a_session
        except Exception as e:
            if current_task:
                await _update_task(str(e), 0)
            a_session = anext(get_db("main_task"))
            session = await a_session

        finally:
            stop_update_task_continuously()
            if current_task and current_task.task_stat != 1:
                await _update_task(stat=0)
            await asyncio.sleep(5)


if __name__ == "__main__":
    print("Detect ship")
    parser = argparse.ArgumentParser()
    parser.add_argument("task_type", type="string", default="object_eo")
    asyncio.run(async_main())
