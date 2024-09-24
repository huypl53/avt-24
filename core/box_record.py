import math
from typing import List

import cv2
import numpy as np


class Movement(dict):
    def __init__(self, displacement: float, rotaion_shift: float) -> None:
        self.displacement = displacement
        self.rotation_shift = rotaion_shift
        # dict.__init__(self, displacement=displacement, rotation_shift=rotaion_shift)
        # mapping = {"displacement": displacement, "rotation_shift": rotaion_shift}
        mapping = self.__dict__
        super().__init__(mapping)


class BoxRecord(dict):
    def __init__(
        self,
        cate_id: int,
        steps_num: int,
        start_step: int = 0,
    ) -> None:
        self.cate_id = cate_id
        self.steps_num = steps_num
        self.last_step: int = 0
        self.history: List[Movement | None] = [None] * (steps_num - 1)
        self.longest_history: List[Movement] = []
        self.records: List[Box | None] = [None] * steps_num
        self.start_step = start_step
        self.longest_sequence: List[Box] = []
        self.max_start_i, self.max_end_i = 0, 0
        self.__update()

    def check_new_target(self, target: "Box", step: int, save=True) -> bool:
        _last_record = self.last_record
        # if self.last_step == 0 or step > self.last_step:
        #     self.last_step = step
        if not _last_record:
            if save:
                self.records[self.last_step] = target
            if self.last_step == 0 or step > self.last_step:
                self.last_step = step
            return True
        movement = Box.detect_movement(_last_record, target)
        if not movement:
            return False
        # self.last_record.went_by = True
        if save:
            self.history[step - 1] = movement
            self.records[step] = target
        return True

    def __update(self):
        dict.update(self, self.__dict__)

    @property
    def last_record(self):
        try:
            return self.records[self.last_step]
        except:
            return None

    def update_longest_sequence(self):
        valid_idx = [i for i, r in enumerate(self.records) if r is not None]
        num_valid_records = len(valid_idx)
        if num_valid_records == 0:
            self.max_start_i = 0
            self.max_end_i = 0
            self.longest_sequence = []
            self.longest_history = []
            return
        elif num_valid_records == 1:
            start_i = valid_idx[0]
            self.max_start_i = start_i
            self.max_end_i = start_i + 1
            self.longest_sequence = [self.records[start_i]]
            self.longest_history = [self.history[start_i]]
            return
        start_i, max_start_i = 0, 0
        max_len = 0
        current_len = 0
        for i in range(len(self.history)):
            hist = self.history[i]
            if hist is not None:
                current_len += 1
                if current_len > max_len:
                    max_len = current_len
                    max_start_i = start_i
            else:
                current_len = 0
                start_i = i + 1
        if max_len < 1:
            return
        max_end_i = max_start_i + max_len + 1
        self.max_start_i = max_start_i
        self.max_end_i = max_end_i
        self.longest_sequence = [r for r in self.records[max_start_i:max_end_i] if r]
        self.longest_history = [
            h for h in self.history[max_start_i : max_end_i - 1] if h
        ]
        self.__update()


class BoxMovement(Movement):
    def __init__(
        self,
        pre_box: "Box",
        next_box: "Box",
        displacement: float,
        rotaion_shift: float,
    ) -> None:
        self.pre_box = pre_box
        self.next_box = next_box
        super().__init__(displacement, rotaion_shift)


class Box(dict):
    def __init__(self, x: float, y: float, w: float, h: float, angle: float):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.angle = angle
        self.went_by = False
        super().__init__(self.__dict__)

    @property
    def box(self):
        return (self.x, self.y, self.w, self.h, self.angle)

    def is_moved(self, target: "Box"):
        """Check if box moved to target
        Args:
            target (BoxDetect | Iterable[Union[ float, int ]]): has same shape as self
        Returns:
            _type_: _description_
        """
        movements = detect_list_roatated_movement(
            [self], [target], translation_threshold=3, rotation_threshold=3, iou=0.12
        )
        if len(movements):
            return movements[0]
        else:
            return None

    @staticmethod
    def detect_movement(box1: "Box", box2: "Box"):
        return box1.is_moved(box2)

    def update(self):
        self.__update()

    def __update(self):
        dict.update(self, self.__dict__)


class BoxDetect(Box):
    def __init__(
        self,
        id: str,
        x: float,
        y: float,
        w: float,
        h: float,
        xl: float,
        yl: float,
        wm: float,
        hm: float,
        angle: float,
        im_path: str,
        lb_path: str,
        cls_name: str = "object",
        score: float = 1.0,
    ) -> None:
        self.xl = xl
        self.yl = yl
        self.wm = wm
        self.hm = hm
        self.cls_name = cls_name
        self.angle = angle
        self.score = score
        self._id = id
        self._im_path = im_path
        self.lb_path = lb_path
        super().__init__(x, y, w, h, angle)
        self.__update()

    @property
    def im_path(self):
        return self._im_path

    @property
    def lat_lon_result(self):
        return [
            self.xl,
            self.yl,
            self.wm,
            self.hm,
            self.angle,
            self.score,
        ]

    @im_path.setter
    def im_path(self, v: str):
        self._im_path = v
        self.__update()

    def __update(self):
        dict.update(
            self,
            {
                "id": self._id,
                "path": self._im_path,
                "lb_path": self.lb_path,
                "coords": self.lat_lon_result,
                "class_id": self.cls_name,
            },
        )

    def is_moved(self, target: "BoxDetect"):
        """Check if box moved to target
        Args:
            target (BoxDetect | Iterable[Union[ float, int ]]): has same shape as self
        Returns:
            _type_: _description_
        """
        if self.cls_name != target.cls_name:
            return None
        return super().is_moved(target)


def rotated_iou(box1, box2):
    (x1, y1, w1, h1, a1) = box1  # (center_x, center_y, width, height, angle)
    (x2, y2, w2, h2, a2) = box2
    rect1 = ((x1, y1), (w1, h1), a1)
    rect2 = ((x2, y2), (w2, h2), a2)
    intersection_type, intersection_points = cv2.rotatedRectangleIntersection(
        rect1, rect2
    )
    if intersection_type == cv2.INTERSECT_NONE:
        return 0.0  # No intersection
    intersection_area = cv2.contourArea(intersection_points)
    rect1_area = w1 * h1
    rect2_area = w2 * h2
    union_area = rect1_area + rect2_area - intersection_area
    iou_value = intersection_area / union_area if union_area != 0 else 0
    return iou_value


def detect_list_roatated_movement(
    box_results1: List[Box],
    box_results2: List[Box],
    translation_threshold=10,
    rotation_threshold=5,
    iou=0.3,
):
    """Check if 2 list of rbboxes 2 x  has movement
    Args:
        boxes1 (List[Box]): [ N x [x, y, w, h, angle] ]
        boxes2 (List[Box]): [ N x [x, y, w, h, angle] ]
        translation_threshold (int, optional): _description_. Defaults to 10.
        rotation_threshold (int, optional): _description_. Defaults to 5.
    Returns:
        movements (List[N x Movement])
    """
    movements = []
    for i, result1 in enumerate(box_results1):
        for j, result2 in enumerate(box_results2):
            box1 = result1.box
            box2 = result2.box
            iou_value = rotated_iou(box1, box2)
            if iou_value > iou:  # Adjust threshold as needed
                displacement, rotation_diff = calc_roatated_movement(box1, box2)
                if (
                    displacement > translation_threshold
                    or rotation_diff > rotation_threshold
                ):
                    m = BoxMovement(result1, result2, displacement, rotation_diff)
                    # movements.append((i, j, m))
                    movements.append(m)
    return movements


def calc_roatated_movement(box1, box2):
    """_summary_
    Args:
        box1 ( List[ float ] ): [ N x [x, y, w, h, angle] ]
        box2 ( List[ float ] ): [ N x [x, y, w, h, angle] ]
    Returns:
        displacement: the center shift
        rotation_diff: the angle shift
    """
    (x1, y1, w1, h1, a1) = box1
    (x2, y2, w2, h2, a2) = box2
    displacement = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2).astype(float)
    rotation_diff = abs(a2 - a1)
    return displacement, rotation_diff


if __name__ == "__main__":
    # id, xc, yc, w, h, angle: rad
    images_coords = [
        [
            [0, 3441.4737, 4054.2935, 10.9474, 67.2024, 2.871593],
            [0, 4470.4307, 4464.8228, 17.0408, 77.7743, 3.021593],
        ],
        [
            [0, 3441.4737, 4058.2935, 10.9474, 67.2024, 2.931593],
            [0, 4476.4307, 4461.8228, 17.0408, 77.7743, 3.021593],
            [0, 4510.9, 4039.3462, 15.8, 55.3077, 0.0],
            [0, 4884.6667, 3735.5, 70.0, 13.0, 0.68],
        ],
        [
            [0, 3444.4737, 4065.2935, 10.9474, 67.2024, 2.971593],
            [0, 4480.4307, 4451.8228, 17.0408, 77.7743, 3.021593],
            [0, 4511.8, 4043.5, 8.0, 33.0, 0.0],
            [0, 4831.1667, 4356.5, 15.0, 69.0, 0.0],
            [0, 4892.5156, 3725.7732, 72.9214, 12.0875, 0.68],
        ],
        [
            [0, 3448.4737, 4069.2935, 10.9474, 67.2024, 3.051593],
            [0, 4485.4307, 4447.8228, 17.0408, 77.7743, 3.051593],
            [0, 4839.1667, 4410.5, 15.0, 69.0, 0.0],
            [0, 4898.7088, 3716.6853, 79.0204, 12.7648, 0.68],
        ],
        [
            [0, 3453.4737, 4075.2935, 10.9474, 67.2024, 3.101593],
            [0, 4853.1667, 4464.5, 15.0, 69.0, 2.841593],
            [0, 4905.7088, 3706.6853, 79.0204, 12.7648, 0.68],
        ],
        [
            [0, 3457.4737, 4084.2935, 10.9474, 67.2024, 0.0],
            [0, 4914.7088, 3698.6853, 79.0204, 12.7648, 0.68],
        ],
    ]

    images_boxes = [
        [Box(*box[1:-1], math.degrees(box[-1])) for box in im_coord]
        for im_coord in images_coords
    ]
    records = []

    num_im = len(images_boxes)
    for s in range(num_im - 1):
        im_boxes = images_boxes[s]
        for box in im_boxes:
            if box.went_by:
                continue
            record = BoxRecord(0, num_im, s)
            checked = record.check_new_target(box, s)
            if checked:
                box.went_by = True
                box.update()

            for n_s in range(s + 1, num_im):
                for next_box in images_boxes[n_s]:
                    if next_box.went_by:
                        continue
                    next_moved = record.check_new_target(next_box, n_s)
                    if next_moved:
                        next_box.went_by = True
                        next_box.update()

            record.update_longest_sequence()
            records.append(record)

    # from pprint import pprint
    import json

    log = dict()
    log.update({"records": records})
    # pprint(records)
    new_records = [r for r in records if r.max_start_i > 0]
    change_records = [r for r in records if len(r.longest_history) / num_im]

    # pprint(f"New records: {new_records}")
    # pprint(f"Change records: {change_records}")
    log.update({"New records": new_records})
    log.update({"Change records": change_records})
    json.dump(log, open("./demo_box_record.json", "w"), indent=4)
