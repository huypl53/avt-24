from collections.abc import AsyncGenerator
from typing import Callable

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.setting import settings

SESSION_FACTORY_STORE = dict()

print(f"Connection string: {settings.asyncpg_url.unicode_string()}")
create_engine = lambda: create_async_engine(
    settings.asyncpg_url.unicode_string(),
    future=True,
    # echo="debug",
)

# expire_on_commit=False will prevent attributes from being expired
# after commit.
AsyncSessionFactory = lambda: async_sessionmaker(
    create_engine(),
    autoflush=False,
    expire_on_commit=False,
)()

create_async_session_factory = lambda engine: async_sessionmaker(
    engine,
    autoflush=False,
    expire_on_commit=False,
)


# Dependency
# async def get_db() -> AsyncGenerator[AsyncSession]:
#     async with AsyncSessionFactory() as session:
#         # logger.debug(f"ASYNC Pool: {engine.pool.status()}")
#         yield session


async def get_db(factory_id: str) -> AsyncGenerator[AsyncSession]:
    global SESSION_FACTORY_STORE
    try:
        if factory_id in SESSION_FACTORY_STORE:
            session = SESSION_FACTORY_STORE[factory_id]()
            yield session
        else:
            engine = create_engine()
            new_factory = create_async_session_factory(engine)
            SESSION_FACTORY_STORE[factory_id] = new_factory
            session = new_factory()
            yield session
    finally:
        await session.close()
