from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class DetectionTaskType(Enum):
    SHIP = 5
    CHANGE = 20
    MILITARY = 21

    def __str__(self) -> str:
        # return f"{self.value} for {self.name}"
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return DetectionTaskType[s]
        except KeyError as exc:
            raise ValueError() from exc


SHIP_LABELS = ["tau_ca", "tau_hang", "tau_quan_su"]


class Management(BaseModel):
    task_id: int
    task_type: int  # 1: hiệu chỉnh (Học)
    # 2: tiền xử lý (Long)
    # 3. lọc mây(Sơn, Trí)
    # 4: enhancement (Huy)
    # 5: detection (Huy)
    # 6: classification
    # 7: object finder (Học)
    task_creator: str
    task_param: str
    task_stat: int
    worker_ip: str
    process_id: int
    task_eta: int
    task_output: str  # JSON of model's results
    task_message: str
    task_id_ref: int


class EnhancementParam(BaseModel):
    input_file: str
    gamma: float = 0.4
    out_dir: str = "/data/RASTER_ARCHIVED/"


class EnhancementOutput(BaseModel):
    output_file: str


class IOParam(BaseModel):
    out_dir: str = "/data/DETECTOR_OUTPUT/"


class DetectionParam(IOParam):
    algorithm: str
    config: str
    checkpoint: str
    device: str
    score_thr: float

    patch_sizes: List[int]
    patch_steps: List[int]
    img_ratios: List[float]
    merge_iou_thr: float

    # DetectionTaskType.change
    consecutive_thr: Optional[float] = None  #


class DetectionInputParam(DetectionParam):
    input_files: List[str]


ObjectCategory = dict(
    {
        0: "plane",
        1: "ship",
        2: "storage_tank",
        3: "baseball_diamond",
        4: "tennis_court",
        5: "basketball_court",
        6: "ground_track_field",
        7: "harbor",
        8: "bridge",
        9: "large_vehicle",
        10: "small_vehicle",
        11: "helicopter",
        12: "roundabout",
        13: "soccer_ball_field",
        14: "swimming_pool",
    }
)
# SHIP = 0
# AIRPORT = 1
# AIRPLANE = 2
# INFRASTRUCTURE = 3
# ROAD = 4
# TANK = 5

# Dota dataset


class ExtractedObject(BaseModel):
    id: str
    path: str
    coords: List[float]
    lb_path: Optional[str]
    class_id: Optional[str] = None
