from collections.abc import AsyncGenerator
from typing import Callable

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.setting import settings

print(f"Connection string: {settings.asyncpg_url.unicode_string()}")
engine = lambda: create_async_engine(
    settings.asyncpg_url.unicode_string(),
    future=True,
    # echo="debug",
)

# expire_on_commit=False will prevent attributes from being expired
# after commit.
AsyncSessionFactory = lambda: async_sessionmaker(
    engine(),
    autoflush=False,
    expire_on_commit=False,
)()


# Dependency
async def get_db() -> AsyncGenerator[AsyncSession]:
    async with AsyncSessionFactory() as session:
        # logger.debug(f"ASYNC Pool: {engine.pool.status()}")
        yield session
