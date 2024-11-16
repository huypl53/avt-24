import multiprocessing.synchronize
from typing import Dict, List, Tuple

from sqlalchemy import Select, select, text
from sqlalchemy.engine.row import Row
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.connector import AsyncSessionFactory, get_db

# from app.db.spawn import DbProcess
from app.model.task import TaskMd
from app.schema import DetectionInputParam, DetectionTaskType, ShipEoDetectionParam


class Worker:
    def __init__(
        self, task_type: DetectionTaskType, pre_param_conf: ShipEoDetectionParam
    ) -> None:
        self._task_type = task_type
        self._pre_param_conf = pre_param_conf
        input_params: DetectionInputParam = DetectionInputParam(
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
