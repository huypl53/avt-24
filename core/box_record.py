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

        self.records: List[BoxDetect | None] = [None] * steps_num
        self.start_step = start_step
        self.longest_sequence: List[BoxDetect] = []
        self.max_start_i, self.max_end_i = 0, 0
        self.__update()

    def check_new_target(self, target: "BoxDetect", step: int, save=True) -> bool:
        _last_record = self.last_record

        if self.last_step == 0 or step > self.last_step:
            self.last_step = step
        if not _last_record:
            if save:
                self.records[step] = target
            return True

        movement = BoxDetect.detect_movement(_last_record, target)
        if save:
            self.history[step] = movement
            self.records[step] = target

        if not movement:
            return False
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
        start_i, max_start_i = 0, 0
        max_len = 0
        current_len = 0
        for i in range(self.steps_num):
            record = self.records[i]
            if record is not None:
                current_len += 1
                if current_len > max_len:
                    max_len = current_len
                    max_start_i = start_i
            else:
                current_len = 0
                start_i = i + 1
        max_end_i = max_start_i + max_len - 1
        if max_len < 1:
            return
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
        pre_box: "BoxDetect",
        next_box: "BoxDetect",
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

        super().__init__(self.__dict__)

    @property
    def box(self):
        return (self.x, self.y, self.w, self.h, self.angle)


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
        cate_id: float = 0,
        score: float = 1.0,
    ) -> None:

        self.xl = xl
        self.yl = yl
        self.wm = wm
        self.hm = hm

        self.cate_id = int(cate_id)
        self.angle = angle
        self.score = score

        self._id = id
        self._im_path = ""
        self.went_by = False

        super().__init__(x, y, w, h, angle)
        self.__update()

    @property
    def im_path(self):
        return self._im_path

    @im_path.setter
    def im_path(self, v: str):
        self._im_path = v
        self.__update()

    def __update(self):
        dict.update(self, self.__dict__)

    def is_moved(self, target: "BoxDetect") -> None | Movement:
        """Check if box moved to target

        Args:
            target (BoxDetect | Iterable[Union[ float, int ]]): has same shape as self

        Returns:
            _type_: _description_
        """
        if self.cate_id != target.cate_id:
            return None
        movements = detect_list_roatated_movement(
            [self], [target], translation_threshold=25, rotation_threshold=7
        )
        if len(movements):
            return movements[0]
        else:
            return None

    @staticmethod
    def detect_movement(box1: "BoxDetect", box2: "BoxDetect"):
        return box1.is_moved(box2)


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
    box_results1: List[BoxDetect],
    box_results2: List[BoxDetect],
    translation_threshold=10,
    rotation_threshold=5,
    iou=0.6,
):
    """Check if 2 list of rbboxes 2 x  has movement

    Args:
        boxes1 (List[BoxDetect]): [ N x [x, y, w, h, angle] ]
        boxes2 (List[BoxDetect]): [ N x [x, y, w, h, angle] ]
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

    displacement = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    rotation_diff = abs(a2 - a1)
    return displacement, rotation_diff


if __name__ == "__main__":
    boxes_image1 = [
        (100, 100, 50, 50, 30),
        (200, 200, 50, 50, 45),
    ]  # (x, y, width, height, angle)
    boxes_image2 = [
        (110, 105, 50, 50, 35),
        (210, 195, 50, 50, 50),
    ]  # Slight movement and rotation

    # movements = detect_list_roatated_movement(boxes_image1, boxes_image2)

    # if movements:
    #     for i, j, displacement, rotation_diff in movements:
    #         print(
    #             f"Object {i} in image 1 moved to position {j} in image 2 with displacement {displacement:.2f} and rotation change {rotation_diff:.2f} degrees"
    #         )
    # else:
    #     print("No significant movements detected.")
