from typing import Optional

from pydantic import BaseModel


class Management(BaseModel):
    task_id: int
    task_type: int  # 1: hiệu chỉnh (Học)
    # 2: tiền xử lý (Long)
    # 3. lọc mây(Sơn, Trí)
    # 4: enhancement (Huy)
    # 5: detection (Huy)
    # 6: classification
    # 7: object finder (Học)
    # 8:
    task_creator: str
    task_param: str
    task_stat: int
    worker_ip: str
    process_id: int
    task_eta: int
    task_output: str  # JSON of model's results
    task_message: str


class EnhancementParam(BaseModel):
    input_file: str
    gamma: float = 0.4
    out_dir: str = "/data/enhancement/"


class EnhancementOutput(BaseModel):
    output_file: str
