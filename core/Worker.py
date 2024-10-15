from app.db.connector import get_db
from app.schema import TaskParam, DetectionTaskType

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
    DetectionTaskParam,
    TaskParam,
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


class Worker:
    def __init__(self, task_type: DetectionTaskType, pre_param_conf: TaskParam) -> None:
        self._task_type = task_type
        self._pre_param_conf = pre_param_conf
        input_params: DetectionTaskParam = DetectionTaskParam(
            **pre_param_conf.model_dump(),
            input_files=[""],
        )
        pass

    async def start(self):
        a_session = anext(get_db())
        self.session = await a_session

        stmt_task = (
            select(TaskMd)
            # .where(TaskMd.id == task_id)
            .where(
                TaskMd.task_type == self._task_type.value
            )  # task type of ship detection
            .where(TaskMd.task_stat < 0)
            .order_by(TaskMd.task_stat.desc())
        )
        tasks = await query_tasks_by_stmt(stmt_task, self.session)

        for i, t in enumerate(tasks):

            pending = await wait_for_ref_task(t, self.session)
            if pending:
                continue


async def wait_for_ref_task(t: TaskMd, session: AsyncSession) -> bool:
    if t.task_id_ref and t.task_id_ref != 0:
        # t has to wait to task with id = t.task_id_ref
        stmt_ref_tasks = (
            select(TaskMd)
            .where(TaskMd.id == t.task_id_ref)  # task type of ship detection
            .where(TaskMd.task_stat == 1)
            .order_by(TaskMd.task_stat.desc())
        )
        tasks = await query_tasks_by_stmt(stmt_ref_tasks, session)
    if len(tasks) == 0:
        t.task_message = "Waiting for task id = {}".format(t.task_id_ref)
        await session.commit()
        return True
    return False


async def query_tasks_by_stmt(stmt, session) -> List[TaskMd]:
    results = await session.execute(stmt)
    mapping_results = results.mappings().all()
    tasks: List[TaskMd] = [m["TaskMd"] for m in mapping_results]
    return tasks
