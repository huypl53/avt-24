from typing import Any

from sqlalchemy import DateTime, Float, Integer, String, Text, TypeDecorator, JSON
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.schema import TaskParam
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


class FieldParam(TypeDecorator):
    impl = JSON

    def process_bind_param(self, value: TaskParam, dialect):
        if value is None:
            return {}
        return value.model_dump_json()

    def process_result_value(self, value: str, dialect):
        if value is not None:
            return TaskParam.model_validate_json(value)
        return None


class TaskMd(BaseMd):
    __tablename__ = "avt_task"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    task_type: Mapped[int] = mapped_column(Integer)
    creator: Mapped[str] = mapped_column(String)
    # task_param: Mapped[str] = mapped_column(String)
    task_param: Mapped[str] = mapped_column(FieldParam)

    task_stat: Mapped[int] = mapped_column(Integer)
    worker_ip: Mapped[str] = mapped_column(String)
    process_id: Mapped[int] = mapped_column(Integer)
    task_eta: Mapped[int] = mapped_column(Integer)
    task_output: Mapped[str] = mapped_column(String)
    task_message: Mapped[str] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column(String)
    updated_at: Mapped[str] = mapped_column(DateTime(timezone=False))
    user_id: Mapped[int] = mapped_column(Integer)
    task_id_ref: Mapped[int] = mapped_column(Integer)


class AdsbMd(BaseMd):
    __tablename__ = "avt_adsb_data"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    lng: Mapped[float] = mapped_column(Float)
    lat: Mapped[float] = mapped_column(Float)
    width: Mapped[float] = mapped_column(Float)
    height: Mapped[float] = mapped_column(Float)
    cog: Mapped[float] = mapped_column(Float)
