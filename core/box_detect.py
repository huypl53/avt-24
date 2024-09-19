from core.movement import Movement


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


from typing import List

import cv2
import numpy as np

from core.box_detect import BoxDetect
from core.movement import BoxMovement, Movement


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
