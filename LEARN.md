# How did I design this project

> Keywords: multi-processing, async, mmdetection, FTP, sqlalchemy, postgre

pass task type to args
---

load tasks by type

- separeated process updates task stat

---

## TaskManager

receive task_type, loop continuously to load task by type
create TaskWorker, start it, wait for its WorkerResult

### TaskWorker
>
> own a new TaskStatUpdater

- param(TaskParamModel)
- update_config():
- on_start():
  - check if task in queue, waiting for parent task
  - start TaskStatUpdater
- on_error():
  - stop TaskStatUpdater
- on_restart():
  - restart TaskStatUpdater

## TaskStatUpdater
>
> update task stat every second

- init(task_id)

## TaskParamModel(BaseModel)
>
> custom of pydantic BaseModel for additional methods

- init(path?, )

### WorkerResult

serializable

### WorkerState

- msg[str]
- status: success | failed | queued

### WorkerSegement

### WorkerDetection

### Image

### ModelBank
