from collections.abc import AsyncGenerator
from typing import Callable

import psycopg2
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.setting import settings

print(f"Connection string: {settings.asyncpg_url.unicode_string()}")
engine = create_async_engine(
    settings.asyncpg_url.unicode_string(),
    future=True,
    # echo="debug",
)

# expire_on_commit=False will prevent attributes from being expired
# after commit.
AsyncSessionFactory = async_sessionmaker(
    engine,
    autoflush=False,
    expire_on_commit=False,
)


# Dependency
async def get_db() -> AsyncGenerator[AsyncSession]:
    async with AsyncSessionFactory() as session:
        # logger.debug(f"ASYNC Pool: {engine.pool.status()}")
        yield session


def query_sync(callback: Callable):
    try:
        host_only, port = settings.POSTGRES_HOST.rsplit(":", 1)
        connection = psycopg2.connect(
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
            host=host_only,
            port=port,
            database=settings.POSTGRES_DB,
        )

        print("Using Python variable in PostgreSQL select Query")
        cursor = connection.cursor()

        callback(cursor)
        # postgreSQL_select_Query = "select * from mobile where id = %s"
        # cursor.execute(postgreSQL_select_Query, (mobileID,))
        # mobile_records = cursor.fetchall()
        # for row in mobile_records:
        #     print(
        #         "Id = ",
        #         row[0],
        #     )
        #     print("Model = ", row[1])
        #     print("Price  = ", row[2])

    except (Exception, psycopg2.Error) as error:
        err_msg = f"Error fetching data from PostgreSQL table: { error}"
        raise err_msg

    finally:
        # closing database connection
        if connection:
            cursor.close()
            connection.close()
            print("psycopg2 connection is closed \n")
