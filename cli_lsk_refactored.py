import asyncio
import traceback
from typing import List
from sqlalchemy.exc import InterfaceError, OperationalError
import logging

from app.schema import DetectionInputParam, DetectionTaskType
from app.model.task import TaskMd
from task_manager import TaskManager
from image_processor import ImageProcessor
from logger import get_main_logger

logger = get_main_logger(
    __name__, log_file="./logs/cli_lsk_refactored.log", level=logging.INFO
)


class LSKProcessor:
    """Main processor that coordinates task management and image processing."""

    def __init__(self):
        self.task_manager = TaskManager()
        self.image_processor = ImageProcessor()
        self.avail_task_types = [
            DetectionTaskType.SHIP,
            # DetectionTaskType.CHANGE,
            # DetectionTaskType.MILITARY,
        ]
        self._num_task_types = len(self.avail_task_types)
        self._i = 0

    async def initialize(self):
        """Initialize the processor."""
        await self.task_manager.initialize_session()

    async def cleanup(self):
        """Clean up resources."""
        self.task_manager.cleanup()
        self.image_processor.cleanup_temp_files()

    def get_next_task_type(self) -> DetectionTaskType:
        """Get next task type in rotation."""
        task_type = self.avail_task_types[self._i % self._num_task_types]
        self._i += 1
        if self._i >= self._num_task_types:
            self._i = 0
        return task_type

    async def process_single_task(self, task_type: DetectionTaskType):
        """Process a single task of the given type."""
        try:
            # Fetch pending tasks
            tasks = await self.task_manager.fetch_pending_tasks(task_type)

            for task in tasks:
                await self._process_task(task, task_type)

        except (InterfaceError, OperationalError) as e:
            await self.task_manager.handle_database_error(e)
        except Exception as e:
            logger.error(f"Unexpected error in process_single_task: {e}")
            logger.error(traceback.format_exc())

    async def _process_task(self, task: TaskMd, task_type: DetectionTaskType):
        """Process a single task."""
        self.task_manager.current_task = task

        try:
            # Check dependent task
            if not await self.task_manager.check_dependent_task(task):
                msg = f"Waiting for task id = {task.task_id_ref}"
                await self.task_manager.mark_task_failed(task, msg)
                return

            # Mark task as processing
            await self.task_manager.mark_task_processing(task)

            # Validate task parameters
            is_valid, error_msg, input_param_dict = (
                await self.task_manager.validate_task_params(task)
            )
            if not is_valid:
                await self.task_manager.mark_task_failed(task, error_msg)
                return

            # Start background task update process
            self.task_manager.start_task_update_process(task, task_type)

            # Load task configuration
            pre_param_conf = self.task_manager.load_task_config(task_type)
            if not pre_param_conf:
                await self.task_manager.mark_task_failed(
                    task, "Failed to load task configuration"
                )
                return

            # Update model parameters
            input_params = self.image_processor.update_model_params(
                input_param_dict, pre_param_conf
            )
            self.image_processor.set_input_params(input_params)

            # Update task parameters
            task.task_param = input_params.model_dump_json(exclude_none=True)
            await self.task_manager.update_task_info(task)

            # Process images
            results = await self._process_images(task, input_params)

            # Mark task as successful
            await self.task_manager.mark_task_success(task, results)

            logger.info(f"Process task id = {task.id} successfully")

        except Exception as e:
            logger.error(f"Error processing task {task.id}: {e}")
            logger.error(traceback.format_exc())
            await self.task_manager.mark_task_failed(task, str(e))
        finally:
            # Clean up task update process
            self.task_manager.stop_task_update_process()
            self.image_processor.cleanup_temp_files()

    async def _process_images(
        self, task: TaskMd, input_params: DetectionInputParam
    ) -> List:
        """Process all images for a task."""
        detect_results = []
        seg_runway_results = []

        for im_th, image_path in enumerate(input_params.input_file):
            image_id = image_path

            # Process image
            _, success = await self.image_processor.process_image(image_path, task.id)
            if not success:
                continue

            # Perform inference
            classes_results, success = await self.image_processor.infer_image()
            if success and classes_results is not None and len(classes_results):
                # Process detection results
                image_detect_results = self.image_processor.process_detection_results(
                    classes_results, image_id, im_th
                )
                if input_params.detect_time:
                    for result in image_detect_results:
                        result.detect_time = input_params.detect_time
                detect_results.append(
                    {
                        "image_id": image_id,
                        "detections": [r.model_dump() for r in image_detect_results],
                    }
                )

            # Process runway segmentation
            # TODO: fix runway error
            # runway_results = self.image_processor.process_runway_segmentation(image_id, im_th)
            # seg_runway_results.append({
            #     "image_id": image_id,
            #     "runway": runway_results
            # })

        # Combine results
        output_dict = [image_result["detections"] for image_result in detect_results]
        output_dict += [image_result["runway"] for image_result in seg_runway_results]

        if not self.image_processor.task_infer_image_success:
            raise Exception("Task inference failed!")

        return output_dict

    async def run(self):
        """Main run loop."""
        await self.initialize()

        try:
            while True:
                task_type = self.get_next_task_type()
                await self.process_single_task(task_type)
                await asyncio.sleep(5)

        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
        except Exception as e:
            logger.error(f"Fatal error in main loop: {e}")
            logger.error(traceback.format_exc())
        finally:
            await self.cleanup()


async def async_main():
    """Main async function."""
    processor = LSKProcessor()
    await processor.run()


if __name__ == "__main__":
    print("Detect ship - Refactored Version")
    asyncio.run(async_main())
