# LSK CLI Refactoring

This document explains the refactored structure that separates task management and image processing concerns for better debugging and maintenance.

## Overview

The original `cli_lsk_v1.py` had tightly coupled task management and image processing logic. The refactored version separates these concerns into three main components:

1. **TaskManager** (`task_manager.py`) - Handles all database operations and task state management
2. **ImageProcessor** (`image_processor.py`) - Handles all image processing, model management, and inference
3. **LSKProcessor** (`cli_lsk_refactored.py`) - Coordinates between the two modules

## File Structure

```
├── cli_lsk_v1.py              # Original monolithic file
├── task_manager.py            # New: Task management module
├── image_processor.py         # New: Image processing module
├── cli_lsk_refactored.py      # New: Main coordinator
├── test_refactored_modules.py # New: Test script
└── REFACTORING_README.md      # This file
```

## Module Responsibilities

### TaskManager (`task_manager.py`)

**Responsibilities:**
- Database session management
- Task fetching and validation
- Task state updates
- Background task status updates
- Parameter parsing and validation
- Configuration loading

**Key Methods:**
- `fetch_pending_tasks()` - Get tasks from database
- `validate_task_params()` - Validate task parameters
- `mark_task_processing()` - Update task status
- `mark_task_success()` / `mark_task_failed()` - Final status updates
- `start_task_update_process()` - Background status updates

### ImageProcessor (`image_processor.py`)

**Responsibilities:**
- Model loading and management
- Image loading and preprocessing
- Inference execution
- Result processing and formatting
- Memory management
- Temporary file cleanup

**Key Methods:**
- `update_model_params()` - Update and reload models if needed
- `process_image()` - Load and prepare image
- `infer_image()` - Run inference
- `process_detection_results()` - Process detection outputs
- `process_runway_segmentation()` - Process runway segmentation
- `handle_memory_error()` - Handle out-of-memory situations

### LSKProcessor (`cli_lsk_refactored.py`)

**Responsibilities:**
- Orchestrating task and image processing
- Error handling and recovery
- Main application loop
- Resource cleanup

**Key Methods:**
- `process_single_task()` - Process one task type
- `_process_task()` - Handle individual task
- `_process_images()` - Process all images for a task
- `run()` - Main application loop

## Benefits of Refactoring

### 1. **Easier Debugging**
- Each module has focused responsibilities
- Can test modules independently
- Clear separation of concerns
- Dedicated logging for each module

### 2. **Better Maintainability**
- Smaller, focused classes
- Clear interfaces between modules
- Easier to modify individual components
- Reduced code duplication

### 3. **Improved Testing**
- Can unit test each module separately
- Mock dependencies easily
- Test specific functionality in isolation

### 4. **Enhanced Error Handling**
- Module-specific error handling
- Better error isolation
- Clearer error messages
- Graceful degradation

## Usage Examples

### Running the Refactored Version

```bash
python cli_lsk_refactored.py
```

### Testing Individual Modules

```bash
python test_refactored_modules.py
```

### Debugging Task Management

```python
from task_manager import TaskManager

async def debug_task_validation():
    task_manager = TaskManager()
    await task_manager.initialize_session()
    
    # Test parameter validation
    test_params = {"image_type": "EO", "input_file": ["test.tif"]}
    is_valid, error, params = await task_manager.validate_task_params_mock(test_params)
    print(f"Valid: {is_valid}, Error: {error}")
```

### Debugging Image Processing

```python
from image_processor import ImageProcessor

def debug_model_loading():
    processor = ImageProcessor()
    
    # Test model parameter updates
    config = EODetectionParam(...)
    input_params = processor.update_model_params(input_dict, config)
    print(f"Model reload needed: {processor.reload_model}")
```

## Migration Guide

### From Original to Refactored

1. **Replace the main file:**
   ```bash
   # Instead of
   python cli_lsk_v1.py
   
   # Use
   python cli_lsk_refactored.py
   ```

2. **Update imports if needed:**
   ```python
   # Old: All functionality in one file
   from cli_lsk_v1 import async_main
   
   # New: Use specific modules
   from task_manager import TaskManager
   from image_processor import ImageProcessor
   from cli_lsk_refactored import LSKProcessor
   ```

3. **Testing:**
   ```python
   # Old: Hard to test individual components
   # New: Easy to test modules independently
   await test_task_manager()
   await test_image_processor()
   ```

## Configuration

The refactored version uses the same configuration files as the original:
- `./config/ship.json`
- `./config/change.json`
- `./config/military.json`

## Logging

Each module has its own log file:
- `./logs/task_manager.log` - Task management operations
- `./logs/image_processor.log` - Image processing operations
- `./logs/cli_lsk_refactored.log` - Main coordinator operations

## Error Handling

The refactored version provides better error isolation:

1. **Database errors** are handled in TaskManager
2. **Model/GPU errors** are handled in ImageProcessor
3. **Integration errors** are handled in LSKProcessor

Each module can recover independently without affecting the others.

## Performance Considerations

- Models are shared between tasks when possible
- Database connections are reused
- Memory is managed more efficiently
- Background processes are properly cleaned up

## Future Enhancements

With the separated structure, it's easier to add:

1. **Parallel processing** - Process multiple tasks simultaneously
2. **Different backends** - Support different database systems
3. **Model caching** - Cache models for better performance
4. **Monitoring** - Add detailed metrics and monitoring
5. **API endpoints** - Expose functionality via REST API

## Troubleshooting

### Common Issues

1. **Module import errors:**
   - Ensure all dependencies are installed
   - Check Python path includes the project directory

2. **Database connection issues:**
   - Check database configuration
   - Verify network connectivity
   - Review TaskManager logs

3. **Model loading issues:**
   - Check GPU availability
   - Verify model files exist
   - Review ImageProcessor logs

4. **Memory issues:**
   - Monitor GPU memory usage
   - Check for memory leaks
   - Use `handle_memory_error()` method

### Debug Mode

Enable debug logging by modifying the logger configuration in each module:

```python
logger = get_main_logger(__name__, log_file="./logs/debug.log", level=logging.DEBUG)
``` 