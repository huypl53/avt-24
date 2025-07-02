#!/usr/bin/env python3
"""
Test script to demonstrate the separated task manager and image processor modules.
This shows how the modules can be used independently for debugging and testing.
"""

import asyncio
import json
from typing import Dict, List
from app.schema import DetectionTaskType, EODetectionParam
from logger import logger
from task_manager import TaskManager
from image_processor import ImageProcessor
import cv2


async def test_task_manager():
    """Test the task manager functionality."""
    print("=== Testing Task Manager ===")

    task_manager = TaskManager()

    # Test parameter parsing
    test_param_str = '{"image_type": "EO", "input_file": ["test1.tif", "test2.tif"], "score_thr": 0.5}'
    parsed_params = task_manager.parse_param_dict(test_param_str)
    print(f"Parsed parameters: {parsed_params}")

    # Test task configuration loading
    config = task_manager.load_task_config(DetectionTaskType.SHIP)
    if config:
        print(f"Loaded ship config: {type(config)}")

    # Test parameter validation
    valid_params = {"image_type": "EO", "input_file": ["test.tif"]}
    is_valid, error_msg, params = await task_manager.validate_task_params_mock(
        valid_params
    )
    print(f"Valid params: {is_valid}, Error: {error_msg}")

    invalid_params = {"image_type": "SAR", "input_file": ["test.tif"]}
    is_valid, error_msg, params = await task_manager.validate_task_params_mock(
        invalid_params
    )
    print(f"Invalid params: {is_valid}, Error: {error_msg}")

    print("Task Manager tests completed!\n")


async def test_image_processor():
    """Test the image processor functionality."""
    print("=== Testing Image Processor ===")

    image_processor = ImageProcessor()

    # Test model parameter update
    test_config = EODetectionParam(
        out_dir="./output",
        algorithm="phat_hien_tau_eo",
        config="/workspace/mmrotate/configs/s2anet/s2anet_r50_fpn_fp16_1x_dota_le135.py",
        checkpoint="/workspace/avt-detection/eo/s2anet_r50_fpn_fp16_1x_dota_le135-5cac515c.pth",
        device="cuda:0",
        score_thr=0.3,
        patch_sizes=[1024],
        patch_steps=[824],
        img_ratios=[1],
        merge_iou_thr=0.1,
        input_file=[""],
        image_type="EO",
        runway_config="/workspace/avt-detection/eo/runway_seg_config.py",
        runway_ckpt="/workspace/avt-detection/eo/runway_seg_ckpt.pth",
        runway_min_length=500,
    )

    input_param_dict = {
        "image_type": "EO",
        "input_file": ["tmp/quang_ninh_1m_part_1.tif"],
        "score_thr": 0.7,  # Different from config
        "device": "cuda:0",
    }

    try:
        input_params = image_processor.update_model_params(
            input_param_dict, test_config
        )
        image_processor.set_input_params(input_params)
        print(f"Updated input params: score_thr={input_params.score_thr}")
        print(f"Model reload needed: {image_processor.reload_model}")
    except Exception as e:
        print(f"Model update test failed (expected without actual models): {e}")

    detect_results = []
    seg_runway_results = []

    for im_th, image_path in enumerate(input_param_dict["input_file"]):
        image_id = image_path

        # Process image
        # _, success = await image_processor.process_image(image_path, im_th)
        # if not success:
        #     continue

        image_processor.im = cv2.imread(image_path)
        image_processor.tmp_im_path = image_path
        image_processor.current_task_id = 1
        # Perform inference
        classes_results, success = await image_processor.infer_image()
        if success and classes_results is not None and len(classes_results):
            logger.info(f"Infer success, found {len(classes_results)} bboxes")
            # Process detection results
            # image_detect_results = image_processor.process_detection_results(
            #     classes_results, image_id, im_th
            # )
            # detect_results.append({
            #     "image_id": image_id,
            #     "detections": image_detect_results
            # })
    print(f"Detect results: {detect_results}")

    for im_th, image_path in enumerate(input_param_dict["input_file"]):
        print(f"Processing image {im_th} of {len(input_param_dict['input_file'])}")
        image_id = image_path

        # Process runway segmentation
        runway_results = image_processor.process_runway_segmentation(image_id, im_th)
        seg_runway_results.append({"image_id": image_id, "runway": runway_results})

    print(f"Seg runway results: {seg_runway_results}")

    # Test memory error handling
    image_processor.handle_memory_error()
    print("Memory error handling test completed")
    print("Image Processor tests completed!\n")


async def test_integration():
    """Test integration between task manager and image processor."""
    print("=== Testing Integration ===")

    task_manager = TaskManager()
    image_processor = ImageProcessor()

    # Simulate task processing workflow
    print("1. Task Manager validates parameters")
    valid_params = {"image_type": "EO", "input_file": ["test.tif"]}
    is_valid, error_msg, params = await task_manager.validate_task_params_mock(
        valid_params
    )

    if is_valid:
        print("2. Image Processor updates model parameters")
        test_config = EODetectionParam(
            out_dir="./output",
            algorithm="phat_hien_tau_eo",
            config="/workspace/mmrotate/configs/s2anet/s2anet_r50_fpn_fp16_1x_dota_le135.py",
            checkpoint="/workspace/avt-detection/eo/s2anet_r50_fpn_fp16_1x_dota_le135-5cac515c.pth",
            device="cuda:0",
            score_thr=0.3,
            patch_sizes=[1024],
            patch_steps=[824],
            img_ratios=[1],
            merge_iou_thr=0.1,
            input_file=[""],
            image_type="EO",
            runway_config="/workspace/avt-detection/eo/runway_seg_config.py",
            runway_ckpt="/workspace/avt-detection/eo/runway_seg_ckpt.pth",
            runway_min_length=500,
        )
        try:
            input_params = image_processor.update_model_params(params, test_config)
            image_processor.set_input_params(input_params)
            print("3. Integration successful!")
        except Exception as e:
            print(f"3. Integration failed (expected without models): {e}")

    print("Integration tests completed!\n")


# Add mock method to TaskManager for testing
async def validate_task_params_mock(self, input_param_dict):
    """Mock version of validate_task_params for testing."""
    if "image_type" not in input_param_dict:
        return False, "<image_type> field is required!", {}

    if input_param_dict["image_type"] != "EO":
        return False, "Only EO image type is supported", {}

    if "input_file" not in input_param_dict:
        return False, "<input_file> field is required!", {}

    return True, "", input_param_dict


# Add the mock method to TaskManager class
TaskManager.validate_task_params_mock = validate_task_params_mock


async def main():
    """Run all tests."""
    print("Starting module tests...\n")

    # await test_task_manager()
    await test_image_processor()
    # await test_integration()

    print("All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())

