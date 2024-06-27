from typing import Any

from sqlalchemy import Integer, String, Text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from log import logger


class BaseMd(DeclarativeBase):
    id: Any
    __name__: str

    async def save(self, db_session: AsyncSession):
        """

        :param db_session:
        :return:
        """
        try:
            db_session.add(self)
            return await db_session.commit()
        except SQLAlchemyError as ex:
            logger.error(f"{ex}")

    async def delete(self, db_session: AsyncSession):
        """

        :param db_session:
        :return:
        """
        try:
            await db_session.delete(self)
            await db_session.commit()
            return True
        except SQLAlchemyError as ex:
            logger.error(f"{ex}")

    async def update(self, db: AsyncSession, **kwargs):
        """

        :param db:
        :param kwargs
        :return:
        """
        try:
            for k, v in kwargs.items():
                setattr(self, k, v)
            return await db.commit()
        except SQLAlchemyError as ex:
            logger.error(f"{ex}")


class TaskMd(BaseMd):
    __tablename__ = "avt_task"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    type: Mapped[int] = mapped_column(Integer)
    creator: Mapped[str] = mapped_column(String)
    task_param: Mapped[str] = mapped_column(String)
    task_stat: Mapped[int] = mapped_column(Integer)
    worker_ip: Mapped[str] = mapped_column(String)
    process_id: Mapped[int] = mapped_column(Integer)
    task_eta: Mapped[int] = mapped_column(Integer)
    task_output: Mapped[str] = mapped_column(String)
    task_message: Mapped[str] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column(String)
    updated_at: Mapped[str] = mapped_column(String)
    user_id: Mapped[int] = mapped_column(Integer)
