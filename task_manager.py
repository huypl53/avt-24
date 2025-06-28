import asyncio
import json
import multiprocessing
import multiprocessing.synchronize
import os
import re
import traceback
from datetime import datetime
from typing import Dict, List, Optional
from sqlalchemy import select, text
from sqlalchemy.exc import InterfaceError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from app.db.connector import get_db
from app.model.task import TaskMd
from app.schema import DetectionTaskType, EODetectionParam
from logger import get_main_logger

logger = get_main_logger(__name__, log_file="./logs/task_manager.log", level=logging.INFO)


class TaskManager:
    """Manages task operations including fetching, updating, and state management."""
    
    def __init__(self):
        self.session: Optional[AsyncSession] = None
        self.update_process: Optional[multiprocessing.Process] = None
        self.stop_event: Optional[multiprocessing.synchronize.Event] = None
        self.current_task: Optional[TaskMd] = None
    
    async def initialize_session(self):
        """Initialize database session."""
        if not self.session:
            a_session = anext(get_db("main_task"))
            self.session = await a_session
    
    async def close_session(self):
        """Close database session."""
        if self.session and self.session.is_active:
            await self.session.close()
            self.session = None
    
    async def update_task_info(
        self, 
        task: TaskMd, 
        msg: str = "", 
        task_stat: Optional[int] = None
    ):
        """Update task information in database."""
        if task_stat is not None:
            task.task_stat = task_stat
        if msg:
            task.task_message = msg
        task.updated_at = datetime.now()
        await self.session.commit()
    
    def parse_param_dict(self, param_str: str) -> Dict:
        """Parse task parameters from JSON string."""
        param = json.loads(param_str)
        for k, v in param.items():
            if type(v) != str:
                continue
            if re.search(r'^"\[.*\]"$', v):
                param[k] = v[1:-1]
        return param
    
    def load_task_config(self, task_type: DetectionTaskType) -> Optional[EODetectionParam]:
        """Load task configuration based on task type."""
        match task_type:
            case DetectionTaskType.SHIP:
                config = open("./config/ship.json", "r").read()
                return EODetectionParam.model_validate_json(config)
            case DetectionTaskType.CHANGE:
                config = open("./config/change.json", "r").read()
                return EODetectionParam.model_validate_json(config)
            case DetectionTaskType.MILITARY:
                config = open("./config/military.json", "r").read()
                return EODetectionParam.model_validate_json(config)
            case _:
                return None
    
    async def query_tasks_by_stmt(self, stmt, session) -> List[TaskMd]:
        """Execute query and return task list."""
        results = await session.execute(stmt)
        mapping_results = results.mappings().all()
        tasks: List[TaskMd] = [m["TaskMd"] for m in mapping_results]
        return tasks
    
    async def fetch_pending_tasks(self, task_type: DetectionTaskType) -> List[TaskMd]:
        """Fetch pending tasks from database."""
        stmt_task = (
            select(TaskMd)
            .where(TaskMd.task_type == task_type.value)
            .where(TaskMd.task_stat < 0)
            .order_by(TaskMd.task_stat.desc())
        )
        return await self.query_tasks_by_stmt(stmt_task, self.session)
    
    async def check_dependent_task(self, task: TaskMd) -> bool:
        """Check if dependent task is completed."""
        if not task.task_id_ref or task.task_id_ref == 0:
            return True
        
        stmt_ref_tasks = (
            select(TaskMd)
            .where(TaskMd.id == task.task_id_ref)
            .where(TaskMd.task_stat == 1)
            .order_by(TaskMd.task_stat.desc())
        )
        sub_tasks = await self.query_tasks_by_stmt(stmt_ref_tasks, self.session)
        return len(sub_tasks) > 0
    
    def start_task_update_process(self, task: TaskMd, task_type: DetectionTaskType):
        """Start background process for updating task status."""
        self.stop_task_update_process()
        
        self.stop_event = multiprocessing.Event()
        self.update_process = multiprocessing.Process(
            target=self._update_task_chronologically,
            args=(task.id, self.stop_event, task_type.value),
        )
        self.update_process.start()
    
    def stop_task_update_process(self):
        """Stop background task update process."""
        if self.stop_event:
            self.stop_event.set()
        if self.update_process:
            self.update_process.terminate()
            self.update_process.join()
            self.update_process = None
            self.stop_event = None
    
    def _update_task_chronologically(
        self,
        task_id: int,
        stop_event: multiprocessing.synchronize.Event,
        task_type: int,
        start: int = 2,
        step: int = 1,
    ):
        """Background process for updating task status chronologically."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def run():
            session = None
            query = text(
                f"SELECT * FROM public.avt_task where task_type = {task_type} and id = {task_id}"
            )

            try:
                a_session = anext(get_db("task_stat_update"))
                session = await a_session
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
                logger.error(traceback.format_exc())
            finally:
                if session and session.is_active:
                    await session.close()

        loop.run_until_complete(run())
        loop.close()
    
    async def validate_task_params(self, task: TaskMd) -> tuple[bool, str, Dict]:
        """Validate task parameters and return validation result."""
        input_param_dict = self.parse_param_dict(task.task_param)
        
        if "image_type" not in input_param_dict:
            return False, "<image_type> field is required!", {}
        
        if input_param_dict["image_type"] != "EO":
            return False, "Only EO image type is supported", {}
        
        if "input_file" not in input_param_dict:
            return False, "<input_file> field is required!", {}
        
        return True, "", input_param_dict
    
    async def mark_task_processing(self, task: TaskMd):
        """Mark task as being processed."""
        task.process_id = os.getpid()
        await self.update_task_info(task, "Task is being processed")
    
    async def mark_task_success(self, task: TaskMd, output_dict: List, extra_message: str = ""):
        """Mark task as successfully completed."""
        task.task_output = json.dumps(output_dict)
        task.task_stat = 1
        task.task_message = "\n".join(["Successfully", extra_message])
        await self.update_task_info(task)
    
    async def mark_task_failed(self, task: TaskMd, error_message: str):
        """Mark task as failed."""
        await self.update_task_info(task, error_message, 0)
    
    async def handle_database_error(self, error: Exception):
        """Handle database connection errors."""
        logger.error(f"Database error occurred: {error}")
        await self.close_session()
        await self.initialize_session()
    
    def cleanup(self):
        """Clean up resources."""
        self.stop_task_update_process()
        if self.session:
            asyncio.create_task(self.close_session()) 