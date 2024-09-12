# from log import logger
import asyncio
import json
import multiprocessing
import multiprocessing.synchronize
import os
import re
import traceback
from typing import Dict, List, Tuple
import argparse

import cv2
import numpy as np
from dictdiffer import diff
from mmdet.apis import init_detector
from mmrotate.apis import inference_detector_by_patches
from sqlalchemy import Select, select, text
from sqlalchemy.engine.row import Row
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.connector import AsyncSessionFactory, get_db

# from app.db.spawn import DbProcess
from app.model.task import TaskMd
from app.schema import (
    DETECT_CHANGE_TASK_TYPE,
    DETECT_MILITARY_TASK_TYPE,
    DETECT_SHIP_TASK_TYPE,
    DetectionInputParam,
    DetectionParam,
    ExtractedObject,
)

from app.service.binio import (
    ftpTransfer,
    read_ftp_bin_image,
    write_ftp_image,
    write_text_file,
)
from utils.lsk import crop_rotated_rectangle, xywhr2xyxyxyxy
from utils.raster import latlong2meter, pixel_point_to_lat_long

from log import logger


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


async def async_main(task_type: int):
    # assert len(sys.argv) < 2
    # task_id = int(sys.argv[1])

    input_params: DetectionInputParam = {}
    model = None
    current_task = None
    reload_model = False
    tmp_im_path = ""
    im: np.ndarray = None
    bname: str = ""
    save_dir: str = ""
    pre_param_conf = DetectionParam()  # .model_dump()
    # counter = 0
    while True:
        # counter += 1
        a_session = anext(get_db())
        session = await a_session
        stmt_task = (
            select(TaskMd)
            # .where(TaskMd.id == task_id)
            .where(TaskMd.task_type == task_type)  # task type of ship detection
            .where(TaskMd.task_stat < 0)
            .order_by(TaskMd.task_stat.desc())
        )
        tasks = await query_tasks_by_stmt(stmt_task, session)

        print("----------")
        # db_thread: DbProcess = None
        update_process: multiprocessing.Process = None
        stop_event: multiprocessing.synchronize.Event = None

        def _update_process_func(t: TaskMd):
            nonlocal update_process, stop_event
            if update_process:
                update_process.terminate()
                update_process.join()
            if stop_event:
                stop_event.set()

            stop_event = multiprocessing.Event()
            update_process = multiprocessing.Process(
                target=update_task_chronologically, args=([t.id, stop_event, task_type])
            )

            update_process.start()

        def _update_param(input_param_dict: Dict):
            nonlocal input_params, pre_param_conf, reload_model, model
            input_param_no_file_dict = (
                {k: v for k, v in input_param_dict.items() if k != "input_files"},
            )

            new_params_cnt = (
                False
                if len(input_param_no_file_dict.items()) == 0
                else len(
                    list(
                        diff(
                            input_param_no_file_dict,
                            dict(pre_param_conf),
                        )
                    )
                )
            )
            if new_params_cnt:
                logger.info(
                    f"new_params_cnt: {new_params_cnt}, task: {input_param_dict}"
                )
                reload_model = True
                # pre_conf.update(param_dict)
                pre_param_conf = pre_param_conf.model_validate(input_param_dict)
                input_params = DetectionInputParam.model_validate(
                    {
                        **pre_param_conf.model_dump(),
                        "input_files": input_param_dict["input_files"],
                    }
                )
            if reload_model:
                model = init_detector(
                    input_params.config,
                    input_params.checkpoint,
                    device=input_params.device,
                )
                reload_model = False

        async def _process_image(input_file: str):
            nonlocal bname, save_dir, im, tmp_im_path, input_params
            bname = os.path.basename(input_file).rsplit(".", 1)[0]
            save_dir = os.path.join(input_params.out_dir, bname)
            ftpTransfer.mkdir(save_dir)

            try:
                bin_im = read_ftp_bin_image(input_file)
                if not bin_im:
                    await _update_task(f"Read image failed at {input_file}")
                    return None, False
            except:
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
            for i, t in enumerate(tasks):
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
                    result, success = await _infer_image_params()
                    if not success:
                        continue
                    if not len(result):
                        await _update_task("No detection", 1)
                        continue

                    output = np.array(result[0])  # take first list of boxes from batch

                    # TODO: handle score thresh
                    output = output[output[..., -1] > input_params.score_thr]

                    if not len(output):
                        await _update_task("No detection reachs score thresh", 1)
                        continue
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
                        await _update_task("Read crs from image failed!")
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

                    image_detect_results: List[Dict] = []
                    for i, (p, c) in enumerate(zip(patches, lat_long_coords)):
                        im_id = f"{i:03d}"
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
                        image_detect_results.append(
                            ExtractedObject(
                                id=im_id,
                                path=patch_im_path,
                                coords=coords,
                                lb_path=patch_lb_path,
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
            # if db_thread:
            #     db_thread.stop()
            #     db_thread.join()
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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_type",
        type=int,
        choices=[
            DETECT_SHIP_TASK_TYPE,
            DETECT_CHANGE_TASK_TYPE,
            DETECT_MILITARY_TASK_TYPE,
        ],
        required=True,
        help="Task type",
    )
    args, _ = parser.parse_known_args()
    asyncio.run(async_main(args.task_type))
