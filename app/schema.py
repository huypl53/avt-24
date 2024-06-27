from typing import Optional, List

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


class DetectShipParam(BaseModel):
    input_file: str
    config: str = './lsknet/configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90.py'
    checkpoint: str = './epoch_3_050324.pth'
    device: str = 'cuda:0'

    patch_sizes: List[int] = [1024]
    patch_steps: List[int] = [824]
    img_ratios: List[float] = [1.0]
    merge_iou_thr: float = 0.1
