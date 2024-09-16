from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class DetectionTaskType(Enum):
    ship = 5
    change = 20
    military = 21

    def __str__(self) -> str:
        # return f"{self.value} for {self.name}"
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return DetectionTaskType[s]
        except KeyError as exc:
            raise ValueError() from exc


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


class DetectionInputParam(DetectionParam):
    input_files: List[str]


class ObjectCategory(Enum):
    # SHIP = 0
    # AIRPORT = 1
    # AIRPLANE = 2
    # INFRASTRUCTURE = 3
    # ROAD = 4
    # TANK = 5

    # Dota dataset
    PLANE = 0
    SHIP = 1
    STORAGE_TANK = 2
    BASEBALL_DIAMOND = 3
    TENNIS_COURT = 4
    BASKETBALL_COURT = 5
    GROUND_TRACK_FIELD = 6
    HARBOR = 7
    BRIDGE = 8
    LARGE_VEHICLE = 9
    SMALL_VEHICLE = 10
    HELICOPTER = 11
    ROUNDABOUT = 12
    SOCCER_BALL_FIELD = 13
    SWIMMING_POOL = 14


class ExtractedObject(BaseModel):
    id: str
    path: str
    coords: List[float]
    lb_path: Optional[str]
    class_id: Optional[ObjectCategory.value]
