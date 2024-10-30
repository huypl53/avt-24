import asyncio
import json
import multiprocessing
import multiprocessing.synchronize
import os
import re
import traceback
from datetime import datetime
from typing import Dict, List, Tuple

from core.raster import RasterImage
import cv2
import numpy as np
from dictdiffer import diff
from sqlalchemy import select, text
from sqlalchemy.exc import InterfaceError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.connector import get_db
from app.model.task import TaskMd
from app.schema import (
    ChangeDetectionParam,
    ChangeDetectionParam,
    DetectionTaskType,
)
from app.service.binio import (
    read_ftp_bin_image,
    read_ftp_np_image,
    write_ftp_image,
)
from log import logger
from utils.cfar import CFAR2D, CFARParams
from utils.processing import find_boundary_keypoints
from utils.transform import gen_fft_diff_mask, mask2image

cfar_params = CFARParams(
    guard_cells=(1, 1),  # Smaller guard cells due to matrix size
    training_cells=(5, 5),  # Smaller training cells due to matrix size
    false_alarm_rate=1e-2,  # Higher false alarm rate
    scaling_factor=1.5,  # Lower scaling factor for more detections
)
# Create CFAR detector
cfar_detector = CFAR2D(cfar_params)


async def update_task_info(
    t: TaskMd, msg: str, session: AsyncSession, task_stat: int = 0
):

    if task_stat:
        t.task_stat = task_stat
    if msg:
        t.task_message = msg
    t.updated_at = datetime.now()
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
    session: AsyncSession | None = None,
    start=2,
    step: int = 1,
):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def run(stop_event):
        nonlocal session
        query = text(
            f"SELECT * FROM public.avt_task where task_type = {task_type} and id = {task_id}"
        )

        try:
            if not session:
                a_session = anext(get_db("task_stat_update"))
                session = await a_session
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
            pass
            # await session.commit()
            # if session is not None:
            #     await session.close()

    # asyncio.run(run())
    loop.run_until_complete(run(stop_event))

    loop.close()


async def query_tasks_by_stmt(stmt, session) -> List[TaskMd]:
    results = await session.execute(stmt)
    mapping_results = results.mappings().all()
    tasks: List[TaskMd] = [m["TaskMd"] for m in mapping_results]
    return tasks


# async def async_main(task_type: DetectionTaskType):
async def async_main():
    # assert len(sys.argv) < 2
    # task_id = int(sys.argv[1])

    current_task = None
    bname: str = ""
    task_type = DetectionTaskType.CHANGE

    config = open("./config/change.json", "r").read()
    pre_param_conf = ChangeDetectionParam.model_validate_json(config)

    while True:

        input_params: ChangeDetectionParam = ChangeDetectionParam(
            **pre_param_conf.model_dump()
        )
        extra_mesg = ""
        a_session = anext(get_db("main_task"))
        session = await a_session
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
            nonlocal input_params, pre_param_conf
            if not pre_param_conf:
                return
            input_param_no_file_dict = {
                k: v
                for k, v in input_param_dict.items()
                if k not in ["input_file", "checkpoint", "config"]
            }

            new_params_cnt = len(
                list(
                    diff(
                        input_param_no_file_dict,
                        dict(pre_param_conf),
                    )
                )
            )

            if new_params_cnt:
                logger.info(
                    f"new_params_cnt: {new_params_cnt}, task: {input_param_dict}"
                )
                # pre_conf.update(param_dict)
                pre_param_conf = pre_param_conf.model_copy(
                    update=input_param_no_file_dict
                )
            input_params = ChangeDetectionParam.model_validate(
                {
                    **pre_param_conf.model_dump(),
                    **input_param_no_file_dict,
                    "input_file": input_param_dict["input_file"],
                }
            )

        async def _process_image(input_file: str, return_bin: bool = False) -> Tuple[None | np.ndarray, bool]:
            # nonlocal bname, save_dir, input_params, task_infer_image_success
            # bname = os.path.basename(input_file).rsplit(".", 1)[0]
            # save_dir = os.path.join(input_params.out_dir, bname)
            # ftpTransfer.mkdir(save_dir)

            try:
                bin_im = read_ftp_bin_image(input_file)
                if not bin_im:
                    await _update_task(f"Read image failed at {input_file}")
                    return None, False
            except Exception:
                await _update_task(f"Read image failed at {input_file}")
                return None, False

            if return_bin:
                return bin_im, True
            image = np.asarray(bytearray(bin_im), dtype="uint8")
            im = cv2.imdecode(image, cv2.IMREAD_COLOR)
            return im, True

        async def _update_task(msg: str = "", stat: int | None = None):
            nonlocal session, current_task, extra_mesg
            task_stat = 0
            if "Expected all tensors to be on the same device" in msg:
                pass
            if stat is None:
                if current_task is not None:
                    task_stat = current_task.task_stat
            else:
                task_stat = stat
                if stat == 0:
                    stop_update_task_continuously()
            try:
                if not current_task:
                    return
                await update_task_info(current_task, f'{msg}\n{extra_mesg}' if msg else msg, session, task_stat)
            except:
                stop_update_task_continuously()

        def stop_update_task_continuously():
            nonlocal stop_event, update_process
            if stop_event:
                stop_event.set()
            if update_process:
                update_process.terminate()
                update_process.join()

        # counter += 1
        # session = await AsyncSessionFactory()
        stmt_task = (
            select(TaskMd)
            # .where(TaskMd.id == task_id)
            .where(TaskMd.task_type == task_type.value)  # task type of ship detection
            .where(TaskMd.task_stat < 0)
            .order_by(TaskMd.task_stat.desc())
        )
        tasks = await query_tasks_by_stmt(stmt_task, session)

        print("----------")
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
                    tasks = await query_tasks_by_stmt(stmt_ref_tasks, session)
                    if len(tasks) == 0:
                        msg = "Waiting for task id = {}".format(t.task_id_ref)
                        await _update_task(msg)
                        continue
                    pass
                _update_process_func(t)
                msg = "Task is being processed"
                t.process_id = os.getpid()
                await _update_task(msg)

                input_param_dict = parse_param_dict(t.task_param)
                if "input_file" not in input_param_dict:
                    await _update_task("<input_file> field is requried!", 0)
                    continue

                _update_param(input_param_dict)

                # t.task_param = stringify_dict_list(input_params.model_dump())
                t.task_param = input_params.model_dump_json(exclude_none=True)
                await _update_task()


                t.task_stat = 1
                t.task_message = "\n".join(["Successfully", extra_mesg])
                try:
                    task_params = ChangeDetectionParam.model_validate_json(t.task_param)
                except:
                    await _update_task("Invalid input param", 0)
                    continue
                # image_list: List[Tuple[str | np.ndarray]] = []
                failed_images: List[str] = []
                pre_im: None | np.ndarray = None
                features: List[np.ndarray] = []

                raster_images: List[RasterImage] = []
                try:
                    image_files = task_params.input_file
                    if not len(image_files):
                        _update_task(f"Images path must be provided", 0)
                        continue
                    for im_path in image_files:
                        bin_im, success = await _process_image(im_path, return_bin=True)
                        if not success or bin_im is None:
                            failed_images.append(im_path)
                            continue
                        raster_images.append(RasterImage(bin_im))

                    raster_intersection = RasterImage.find_intersection(raster_images[0], raster_images[1:])
                    for raster_image in raster_images:
                        cropped_im, _ = raster_image.crop_raster(raster_intersection)
                        if not _:
                            continue
                        im = cropped_im.numpy
                        if pre_im is None:
                            pre_im = im
                            continue
                        fft_diff = gen_fft_diff_mask(pre_im, im, 64, 32)
                        features.append(fft_diff)
                        pre_im = im
                    if len(features) == 0:
                        raise ValueError("No raster feature found")
                    features = np.mean(features, axis=0)
                except Exception as e:
                    await _update_task(f"Got error: {e}", 0)

                if len(failed_images):
                    extra_mesg += f' Read image failed at: {";".join(failed_images)}'

                feature_image = mask2image(features)
                top_detections = cfar_detector.get_top_detections(feature_image)
                mask_img = np.zeros_like(fft_diff)

                for row, col, value in top_detections:
                    # TODO: binary mask
                    mask_img[row, col] = 255

                filter_image_path = input_params.mask_file
                if filter_image_path:
                    try:
                        image_filter = read_ftp_np_image(filter_image_path)
                        filter_size = np.array( image_filter.shape[:2][::-1])
                        mask_size = np.array(mask_img.shape[:2][::-1])
                        if not (mask_size == filter_size).all():
                            image_filter = cv2.resize(image_filter, mask_size)
                            extra_mesg += '. Mask filter has different size'
                        if len(image_filter.shape) > 2:
                            image_filter = cv2.cvtColor(image_filter, cv2.COLOR_BGR2GRAY)

                        image_filter = image_filter != 0
                        mask_img = mask_img * image_filter
                    except Exception as e:
                        extra_mesg += f'. Reading mask filter failed at {filter_image_path}'
                        pass

                intersection_np = raster_intersection.numpy
                sized_mask_img = cv2.resize(mask_img, intersection_np.shape[:2][::-1])
                sized_mask_img = cv2.normalize(sized_mask_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                keypoint_list = find_boundary_keypoints(sized_mask_img)
                lat_lon_keypoints = [[raster_intersection.pixel_to_coords(x, y) for x, y in kp] for kp in keypoint_list]
                bname = os.path.basename(image_files[0]).rsplit(".", 1)[0]
                file_path = os.path.join(input_params.out_dir, bname) + '_changes.png'
                write_ftp_image(sized_mask_img, '.png', file_path)

                output_dict = dict({
                    "output_file": file_path,
                    "output": lat_lon_keypoints
                })
                
                t.task_output = json.dumps(output_dict)
                logger.info(f"Process task id = {t.id} successfully")
                stop_update_task_continuously()
                await asyncio.sleep(2)
                await _update_task(stat=1)
        except RuntimeError as e:
            stop_update_task_continuously()
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
            await asyncio.sleep(5)

        print("----------")


if __name__ == "__main__":
    asyncio.run(async_main())
