import asyncio
import json
import multiprocessing
import multiprocessing.synchronize
import os
from typing import Dict, List, Tuple

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
from app.schema import DetectShipParam, ExtractedShip
from app.service.binio import (
    ftpTransfer,
    read_ftp_bin_image,
    write_ftp_image,
    write_text_file,
)
from log import logger
from utils.lsk import crop_rotated_rectangle, xywhr2xyxyxyxy
from utils.raster import latlong2meter, pixel_point_to_lat_long

DETECT_TASK_TYPE = 5


async def update_failed_task(
    t: TaskMd, msg: str, session: AsyncSession, task_stat: int = 0
):

    t.task_stat = task_stat
    t.task_message = msg
    t.task_stat = 0
    await session.commit()


def update_task_chronologically(
    task_id: int,
    stop_event,
    db_session: AsyncSession | None = None,
    start=2,
    step: int = 1,
):
    async def run():
        query = text(
            f"SELECT * FROM public.avt_task where task_type = {DETECT_TASK_TYPE} and id = {task_id}"
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
                        f"update public.avt_task set task_stat = {task_stat} where task_type = {DETECT_TASK_TYPE} and id = {task_id}"
                    )
                )
                await session.commit()
                await asyncio.sleep(step)
        except Exception as e:
            logger.error(e)
        finally:
            # await session.commit()
            await session.close()

    # return run
    asyncio.run(run())


async def async_main():
    # assert len(sys.argv) < 2
    # task_id = int(sys.argv[1])

    params: DetectShipParam = {}
    model = None
    current_task = None
    reload_model = False

    pre_conf = DetectShipParam(input_file="").model_dump()
    # counter = 0
    while True:
        # counter += 1
        a_session = anext(get_db())
        session = await a_session
        stmt = (
            select(TaskMd)
            # .where(TaskMd.id == task_id)
            .where(TaskMd.task_type == DETECT_TASK_TYPE)  # task type of ship detection
            .where(TaskMd.task_stat < 0)
            .order_by(TaskMd.task_stat.desc())
        )
        results = await session.execute(stmt)
        mapping_results = results.mappings().all()
        tasks: List[TaskMd] = [m["TaskMd"] for m in mapping_results]
        if not tasks or len(tasks) < 1:
            stmt = (
                select(TaskMd)
                .where(TaskMd.task_type == DETECT_TASK_TYPE)  # task type of ship detection
                .where(TaskMd.task_stat < 0)
                .where(TaskMd.task_id_ref == 0)
                .order_by(TaskMd.task_stat.desc())
            )
            results = await session.execute(stmt)
            mapping_results = results.mappings().all()
            tasks: List[TaskMd] = [m["TaskMd"] for m in mapping_results]

        print("----------")
        # db_thread: DbProcess = None
        update_process: multiprocessing.Process = None
        stop_event: multiprocessing.synchronize.Event = None
        try:
            for i, t in enumerate(tasks):
                if update_process:
                    update_process.terminate()
                    update_process.join()
                if stop_event:
                    stop_event.set()

                stop_event = multiprocessing.Event()
                update_process = multiprocessing.Process(
                    target=update_task_chronologically, args=([t.id, stop_event])
                )

                update_process.start()

                current_task = t
                # t.task_stat = 2  # task is checked in db_thread
                t.task_message = "Task is being processed"
                t.process_id = os.getpid()
                await session.commit()

                # if i == 1:
                #     break  # update only one
                param_dict = json.loads(t.task_param)
                if "input_file" not in param_dict:
                    await update_failed_task(
                        t, "<input_file> field is requried!", session
                    )
                    continue
                new_params_cnt = len(list(diff(param_dict, dict(params))))
                if new_params_cnt:
                    reload_model = True
                    pre_conf.update(param_dict)
                    params = DetectShipParam(**pre_conf)

                if reload_model:
                    model = init_detector(
                        params.config, params.checkpoint, device=params.device
                    )
                    reload_model = False

                t.task_param = json.dumps(params.model_dump())
                await session.commit()

                bin_im = read_ftp_bin_image(params.input_file)
                if not bin_im:
                    await update_failed_task(
                        t, f"Read image failed at {params.input_file}", session
                    )
                    continue
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
                    await update_failed_task(t, "No detection", session, 1)
                    continue
                output = np.array(result[0])  # take first list of boxes from batch

                # TODO: handle score thresh
                output = output[output[..., -1] > params.score_thr]

                if not len(output):
                    await update_failed_task(
                        t, "No detection reachs score thresh", session, 1
                    )
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
                    await update_failed_task(t, "Read crs from image failed!", session)
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
                    path = os.path.join(save_dir, im_id)
                    patch_lb_path = path + ".txt"
                    patch_im_path = path + ".png"
                    # Box xyxyxyxy
                    # coords = xy.tolist()

                    # Box cx, cy, w, h, angle
                    coords = c.tolist()
                    write_ftp_image(p, ".png", patch_im_path)
                    write_text_file(" ".join([str(i) for i in coords]), patch_lb_path)
                    detect_results.append(
                        ExtractedShip(
                            id=im_id,
                            path=patch_im_path,
                            coords=coords,
                            lb_path=patch_lb_path,
                        ).model_dump()
                    )
                # TODO: xyxyxyxy to real coordinates
                # t.task_output = json.dumps(xyxyxyxy.tolist())
                t.task_output = json.dumps(detect_results)
                t.task_stat = 1
                t.task_message = "Successfully"
                if os.path.isfile(tmp_im_path):
                    os.remove(tmp_im_path)
                logger.info(f"Process task id = {t.id} successfully")

        except Exception as e:
            if current_task:
                await update_failed_task(
                    current_task,
                    str(e),
                    session,
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
    # cli_main()
    asyncio.run(async_main())
