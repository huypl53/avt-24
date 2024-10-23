"""
    generate data from .csv file
"""

import sys
import os

app_path = os.path.dirname(sys.path[0])
sys.path.append(app_path)

from app.db.connector import create_engine
from app.model import BaseMd
import asyncio


from tests.log import logger


async def start_db():
    mode = os.getenv("MODE")
    if not mode or mode != "test":
        logger.error(f"Mode test should be specified, not {mode}")
        return
    engine = create_engine()
    async with engine.begin() as conn:
        await conn.run_sync(BaseMd.metadata.drop_all)
        await conn.run_sync(BaseMd.metadata.create_all)
    # for AsyncEngine created in function scope, close and
    # clean-up pooled connections
    await engine.dispose()


asyncio.run(start_db())
