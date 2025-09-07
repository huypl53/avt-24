from collections.abc import AsyncGenerator
from typing import Callable
import sys
import atexit
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.setting import settings

SESSION_FACTORY_STORE = dict()

print(f"Connection string: {settings.asyncpg_url.unicode_string()}")
create_engine = lambda: create_async_engine(
    settings.asyncpg_url.unicode_string(),
    future=True,
    # echo="debug",
    # pool_size=20,                # Max number of connections in the pool
    max_overflow=30,  # Max additional connections beyond pool_size
    pool_timeout=50,  # Timeout to wait for a connection before raising an error
    pool_recycle=1800,  # Time in seconds to recycle the connection (to prevent stale connections)
    pool_pre_ping=True,  # Check if a connection is alive before using it
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
    except Exception as e:
        print(e)
        pass
    # finally:
    #     await session.close()

# @atexit.register
# def cleanup():
#     global SESSION_FACTORY_STORE
#     for factory_id, factory in SESSION_FACTORY_STORE.items():
#         try:
#             engine = factory.kw["bind"]
#             if engine:
#                 engine.dispose()
#                 print(f"Disposed engine for factory_id: {factory_id}")
#         except Exception as e:
#             print(f"Error disposing engine for factory_id {factory_id}: {e}")
#     SESSION_FACTORY_STORE.clear()
#     print("Cleaned up all session factories and disposed engines.")