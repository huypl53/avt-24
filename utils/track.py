from typing import List
from core.box_detect import BoxDetect
from core.movement import BoxMovement, Movement
import cv2
import numpy as np


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
