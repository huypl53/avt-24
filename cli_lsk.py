import argparse
import asyncio
import json
import multiprocessing
import multiprocessing.synchronize
import os
import re
import traceback
from typing import Dict, List, Tuple

import cv2
import numpy as np
from dictdiffer import diff
from core import Worker
from mmdet.apis import init_detector
from mmrotate.apis import inference_detector_by_patches
from sqlalchemy import Select, select, text
from sqlalchemy.engine.row import Row
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.connector import AsyncSessionFactory, get_db

# from app.db.spawn import DbProcess
from app.model.task import TaskMd
from app.schema import (
    DetectionInputParam,
    DetectionParam,
    DetectionTaskType,
    ExtractedObject,
    ObjectCategory,
)
from app.service.binio import (
    ftpTransfer,
    read_ftp_bin_image,
    write_ftp_image,
    write_text_file,
)
from log import logger
from utils.lsk import crop_rotated_rectangle, xywhr2xyxyxyxy
from utils.raster import latlong2meter, pixel_point_to_lat_long, read_tif_meta


async def update_task_info(
    t: TaskMd, msg: str, session: AsyncSession, task_stat: int = 0
):

    t.task_stat = task_stat
    t.task_message = msg
    t.task_stat = 0
    await session.commit()


def parse_param_dict(param_str: str) -> Dict:
    param = json.loads(param_str)
    for k, v in param.items():
        if type(v) != str:
            continue
        if re.search(r'^"\[.*\]"$', v):
            param[k] = v[1:-1]
    return param


def stringify_dict_list(param: Dict):
    for k, v in param.items():
        if isinstance(v, list):
            param[k] = f'"{json.dumps(v)}"'


def filter_3d_array(array3d: np.ndarray, filter2d: np.ndarray) -> np.ndarray:
    output = np.array(
        [
            [bbox for bbox, mask in zip(class_boxes, bbox_masks) if mask]
            for class_boxes, bbox_masks in zip(array3d, filter2d)
        ]
    )
    return output


def update_task_chronologically(
    task_id: int,
    stop_event,
    task_type: int,
    db_session: AsyncSession | None = None,
    start=2,
    step: int = 1,
):
    async def run():
        query = text(
            f"SELECT * FROM public.avt_task where task_type = {task_type} and id = {task_id}"
        )
        session = db_session
        if not session:
            session = AsyncSessionFactory()

        try:
            results = await session.execute(query)
            result = results.first()
            if not result:
                logger.warning(f"No task for Select: {task_id}")
                return
            task: TaskMd = result
            if not task:
                logger.warning(f"No task for Select: {task_id}")
                return
            task_stat = task.task_stat
            if task_stat is None or task_stat < 0:
                task_stat = start
            while not stop_event.is_set():
                task_stat = task_stat + step
                await session.execute(
                    text(
                        f"update public.avt_task set task_stat = {task_stat} where task_type = {task_type} and id = {task_id}"
                    )
                )
                await session.commit()
                await asyncio.sleep(step)
        except Exception as e:
            logger.error(e)
            logger.error(traceback.format_exc())
        finally:
            # await session.commit()
            await session.close()

    # return run
    asyncio.run(run())


async def query_tasks_by_stmt(stmt, session) -> List[TaskMd]:
    results = await session.execute(stmt)
    mapping_results = results.mappings().all()
    tasks: List[TaskMd] = [m["TaskMd"] for m in mapping_results]
    return tasks


def load_task_config(task_type: DetectionTaskType) -> DetectionParam | None:
    match task_type:
        case DetectionTaskType.SHIP:
            config = open("./config/ship.json", "r").read()
            return DetectionParam.model_validate_json(config)
        case DetectionTaskType.CHANGE:
            config = open("./config/change.json", "r").read()
            return DetectionParam.model_validate_json(config)
        case DetectionTaskType.MILLITARY:
            config = open("./config/military.json", "r").read()
            return DetectionParam.model_validate_json(config)
        case _:
            return None


async def async_main(task_type: DetectionTaskType):
    # assert len(sys.argv) < 2
    # task_id = int(sys.argv[1])

    model = None
    current_task = None
    reload_model = False
    tmp_im_path = ""
    im: np.ndarray = None
    bname: str = ""
    save_dir: str = ""
    pre_param_conf = load_task_config(task_type)
    input_params: DetectionInputParam = DetectionInputParam(
        **pre_param_conf.model_dump(),
        input_files=[""],
    )
    # counter = 0
    while True:
        # counter += 1
        a_session = anext(get_db())
        session = await a_session
        stmt_task = (
            select(TaskMd)
            # .where(TaskMd.id == task_id)
            .where(TaskMd.task_type == task_type.value)  # task type of ship detection
            .where(TaskMd.task_stat < 0)
            .order_by(TaskMd.task_stat.desc())
        )
        tasks = await query_tasks_by_stmt(stmt_task, session)

        print("----------")
        # db_thread: DbProcess = None
        update_process: multiprocessing.Process | None = None
        stop_event: multiprocessing.synchronize.Event | None = None

        def _update_process_func(t: TaskMd):
            nonlocal update_process, stop_event
            if update_process:
                update_process.terminate()
                update_process.join()
            if stop_event:
                stop_event.set()

            stop_event = multiprocessing.Event()
            update_process = multiprocessing.Process(
                target=update_task_chronologically,
                args=([t.id, stop_event, task_type.value]),
            )

            update_process.start()

        def _update_param(input_param_dict: Dict):
            nonlocal input_params, pre_param_conf, reload_model, model
            input_param_no_file_dict = {
                k: v for k, v in input_param_dict.items() if k != "input_files"
            }

            new_params_cnt = len(
                list(
                    diff(
                        input_param_no_file_dict,
                        dict(pre_param_conf),
                    )
                )
            )

            if new_params_cnt or not model:
                logger.info(
                    f"new_params_cnt: {new_params_cnt}, task: {input_param_dict}"
                )
                reload_model = True
                # pre_conf.update(param_dict)
                pre_param_conf = pre_param_conf.model_copy(
                    update=input_param_no_file_dict
                )
                input_params = DetectionInputParam.model_validate(
                    {
                        **pre_param_conf.model_dump(),
                        **input_param_dict,
                    }
                )
            if reload_model:
                model = init_detector(
                    input_params.config,
                    input_params.checkpoint,
                    device=input_params.device,
                )
                reload_model = False

        async def _process_image(input_file: str) -> Tuple[None | np.ndarray, bool]:
            nonlocal bname, save_dir, im, tmp_im_path, input_params
            bname = os.path.basename(input_file).rsplit(".", 1)[0]
            save_dir = os.path.join(input_params.out_dir, bname)
            ftpTransfer.mkdir(save_dir)

            try:
                bin_im = read_ftp_bin_image(input_file)
                if not bin_im:
                    await _update_task(f"Read image failed at {input_file}")
                    return None, False
            except Exception:
                await _update_task(f"Read image failed at {input_file}")
                return None, False
            tmp_im_path = f"./tmp/{bname}.tif"
            open(tmp_im_path, "wb").write(bin_im)

            image = np.asarray(bytearray(bin_im), dtype="uint8")
            im = cv2.imdecode(image, cv2.IMREAD_COLOR)
            return im, True

        async def _update_task(msg: str, stat: int = 0):
            nonlocal session, current_task
            await update_task_info(current_task, msg, session, stat)

        async def _infer_image_params() -> Tuple[np.ndarray | None, bool]:
            nonlocal current_task, model, tmp_im_path, im, input_params

            result = inference_detector_by_patches(
                model,
                im,
                input_params.patch_sizes,
                input_params.patch_steps,
                input_params.img_ratios,
                input_params.merge_iou_thr,
            )  # inference for batch

            return result, True

        try:
            for task_i, t in enumerate(tasks):
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
                    tasks = await query_tasks_by_stmt(stmt_ref_tasks, session)
                    if len(tasks) == 0:
                        t.task_message = "Waiting for task id = {}".format(
                            t.task_id_ref
                        )
                        await session.commit()
                        continue
                    pass
                _update_process_func(t)
                current_task = t
                t.task_message = "Task is being processed"
                t.process_id = os.getpid()
                await session.commit()

                input_param_dict = parse_param_dict(t.task_param)
                if "input_files" not in input_param_dict:
                    await _update_task("<input_files> field is requried!")
                    continue

                _update_param(input_param_dict)

                # t.task_param = stringify_dict_list(input_params.model_dump())
                t.task_param = input_params.model_dump_json()
                await session.commit()
                detect_results = []
                for image_path in input_params.input_files:
                    image_id = image_path
                    _, success = await _process_image(image_path)
                    if not success:
                        continue
                    classes_results, success = await _infer_image_params()
                    if not success:
                        continue
                    if classes_results is None or not len(classes_results):
                        await _update_task("No detection", 1)
                        continue

                    # output = np.array(result[0])  # take first list of boxes from batch

                    # TODO: handle score thresh
                    classes_results = np.array(classes_results)
                    # output = result[:, result[..., -1] > input_params.score_thr]
                    score_mask = classes_results[..., -1] > input_params.score_thr
                    output = np.array(
                        [
                            [
                                bbox
                                for bbox, mask in zip(class_boxes, bbox_masks)
                                if mask
                            ]
                            for class_boxes, bbox_masks in zip(
                                classes_results, score_mask
                            )
                        ]
                    )

                    if not len(output):
                        await _update_task("No detection reachs score thresh", 1)
                        continue
                    xyxyxyxy = xywhr2xyxyxyxy(output)
                    output[..., 4] = np.degrees(output[..., 4])

                    # this is to crop detected patach only
                    cls_rbboxes = [
                        [
                            [
                                int(box[0]),
                                int(box[1]),
                                int(box[2]),
                                int(box[3]),
                                box[4],
                            ]
                            for box in class_bboxes
                        ]
                        for class_bboxes in output
                    ]

                    # Pick only valid boxes that provide rectangle
                    # valid_idx: List[int] = []
                    # patches: List[np.ndarray] = []

                    valid_idx = np.zeros(output.shape[:2])
                    patches = np.zeros_like(valid_idx).tolist()
                    for cls_i, rbboxes in enumerate(cls_rbboxes):
                        for box_i, box in enumerate(rbboxes):
                            patch = crop_rotated_rectangle(
                                im, box
                            )  # patch if None if crop failed
                            if patch is not None:
                                # patches.append(patch)
                                # valid_idx.append(cls_id)
                                patches[cls_i][box_i] = patch
                                valid_idx[cls_i, box_i] = 1

                    # output = output[valid_idx]
                    output = filter_3d_array(output, valid_idx)
                    # xyxyxyxy = xyxyxyxy[valid_idx]
                    xyxyxyxy = filter_3d_array(xyxyxyxy, valid_idx)
                    flat_xy = xyxyxyxy.reshape(-1, 2)

                    # Convert angles to `Bearings maths`
                    output[..., 4] -= 90
                    angles = output[..., 4]
                    output[..., 4][angles > 0] = 360 - angles[angles > 0]
                    output[..., 4][angles < 0] = -angles[angles < 0]

                    try:
                        tif_meta = read_tif_meta(tmp_im_path)
                        lat_long_center = [
                            [
                                pixel_point_to_lat_long(
                                    np.int64(box[..., 0:2]), tif_meta
                                )
                                for box in classes_boxes
                            ]
                            for classes_boxes in output
                        ]
                        lat_long_center = np.array(lat_long_center).reshape(
                            output.shape[:-1], 2
                        )
                    except Exception:
                        await _update_task("Read crs from image failed!")
                        continue

                    latlong_xy = pixel_point_to_lat_long(tmp_im_path, flat_xy)
                    latlong_xyxyxyxy = np.array(latlong_xy).reshape(
                        output.shape[:-1], 4, 2
                    )
                    # lat_long_wh = pixel_point_to_lat_long(tmp_im_path, output[..., 2:4])
                    lat_long_wh = np.array(
                        [
                            [
                                [
                                    latlong2meter(
                                        box[i][1],
                                        box[i][0],
                                        box[i + 1][1],
                                        box[i + 1][0],
                                    )
                                    for i in range(2)
                                ]
                                for box in class_bboxes
                            ]
                            for class_bboxes in latlong_xyxyxyxy
                        ]
                    )

                    # [[lat_c, lon_c, w_meter, h_meter, score, angle_degree],]
                    lat_long_coords = np.concatenate(
                        (lat_long_center, lat_long_wh, output[..., 4:]), axis=-1
                    )

                    image_detect_results: List[Dict] = []
                    # for cls_id, (p, c) in enumerate(zip(patches, lat_long_coords)):
                    for cls_i in range(output.shape[0]):
                        for box_i, (p, c) in enumerate(
                            zip(patches[cls_i], lat_long_coords[cls_i])
                        ):
                            im_id = f"{cls_i:03d}_{box_i:04d}"
                            path = os.path.join(save_dir, im_id)
                            patch_lb_path = path + ".txt"
                            patch_im_path = path + ".png"
                            # Box xyxyxyxy
                            # coords = xy.tolist()

                            # Box cx, cy, w, h, angle
                            coords = c.tolist()
                            write_ftp_image(p, ".png", patch_im_path)
                            write_text_file(
                                " ".join([str(i) for i in coords]), patch_lb_path
                            )

                            class_id = (
                                cls_i
                                if task_type != DetectionTaskType.SHIP
                                else ObjectCategory.SHIP
                            )
                            image_detect_results.append(
                                ExtractedObject(
                                    id=im_id,
                                    path=patch_im_path,
                                    coords=coords,
                                    lb_path=patch_lb_path,
                                    class_id=class_id,
                                ).model_dump()
                            )

                    detect_results.append(
                        {"image_id": image_id, "detections": image_detect_results}
                    )
                t.task_output = json.dumps(detect_results)
                t.task_stat = 1
                t.task_message = "Successfully"
                if os.path.isfile(tmp_im_path):
                    os.remove(tmp_im_path)
                logger.info(f"Process task id = {t.id} successfully")

        except Exception as e:
            if current_task:
                await _update_task(
                    str(e),
                )
        finally:
            await session.commit()
            if update_process:
                update_process.terminate()
                update_process.join()
            if stop_event:
                stop_event.set()

        print("----------")
        await asyncio.sleep(3)
        # await session.close()


# Run this from outter directory
# python ./LSKNet/huge_images_extract.py --dir ./images  --config './LSKNet/configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90.py' --checkpoint './epoch_3_050324.pth' --score-thr 0.5 --save-dir /tmp/ships/

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_type",
        type=lambda v: DetectionTaskType[v],
        choices=list(DetectionTaskType),
        required=True,
        help="Task type",
    )
    args, _ = parser.parse_known_args()
    asyncio.run(async_main(args.task_type))

    # pre_param_conf = load_task_config(args.task_type)
    # worker = Worker()
    # asyncio.run(worker.start(args.task_type, pre_param_conf))
