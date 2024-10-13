import abc
import asyncio
import multiprocessing

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.connector import AsyncSessionFactory
from app.model.task import TaskMd
from log import logger


class Updater(abc.ABC):
    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def start(self):
        pass


class ChronologicalTaskUpdater(Updater):
    def __init__(self) -> None:
        super().__init__()

    def start(self):
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
            # logger.error(traceback.format_exc())
        finally:
            # await session.commit()
            await session.close()

    # return run
    asyncio.run(run())
