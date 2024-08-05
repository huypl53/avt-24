import asyncio
import threading
import time
from collections.abc import AsyncGenerator, Callable

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.connector import AsyncSessionFactory


class DbThread(threading.Thread):
    def __init__(
        self,
        async_sess_maker: async_sessionmaker,
        generator_creator: Callable[[AsyncSession], AsyncGenerator],
    ):
        super().__init__()
        self.__stop_event = threading.Event()
        self.__loop: asyncio.AbstractEventLoop = None
        self.__generator_creator = generator_creator
        self.__session_factory = async_sess_maker

    async def run_transaction(self):
        async with self.__session_factory() as session:
            cnt = 5
            while not self.__stop_event.is_set():
                # gen = self.__generator_creator(session)
                # async for _ in gen():
                #     pass
                await session.execute(
                    f"update public.avt_task set task_stat = {cnt} where task_type = 5 and id = 68"
                )
                await session.commit()
                time.sleep(1)
                cnt += 1

    def run(self):
        self.__loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.__loop)
        self.__loop.run_until_complete(self.run_transaction())
        pass

    def stop(self):
        self.__stop_event.set()
        # if self.__loop:
        #     self.__loop.call_soon_threadsafe(self.__loop.stop)

        self.join()
