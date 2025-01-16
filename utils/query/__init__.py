from app.db.connector import get_db
from app.model.task import TaskMd


from typing import List


async def query_tasks_by_stmt(stmt, session) -> List[TaskMd]:
    results = await session.execute(stmt)
    mapping_results = results.mappings().all()
    tasks: List[TaskMd] = [m["TaskMd"] for m in mapping_results]
    return tasks


async def _update_task(
    task: TaskMd, msg: str = "", session_name="main_task", stat: int | None = None
):
    a_session = anext(get_db(session_name))
    session = await a_session

    task_stat = 0
    if "Expected all tensors to be on the same device" in msg:
        pass
    if stat is None:
        if task is not None:
            task_stat = task.task_stat
    else:
        task_stat = stat
        if stat == 0:
            stop_update_task_continuously()
    try:
        if not task:
            return
        await update_task_info(task, msg, session, task_stat)

        if task_stat:
            task.task_stat = task_stat
        if msg:
            task.task_message = msg
        task.updated_at = datetime.now()
        await session.commit()

    except:
        stop_update_task_continuously()
