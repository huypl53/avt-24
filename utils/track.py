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


def detect_movement_rotated(
    boxes1, boxes2, translation_threshold=10, rotation_threshold=5
):
    movements = []

    for i, box1 in enumerate(boxes1):
        for j, box2 in enumerate(boxes2):
            iou_value = rotated_iou(box1, box2)
            if iou_value > 0.5:  # Adjust threshold as needed
                (x1, y1, w1, h1, a1) = box1
                (x2, y2, w2, h2, a2) = box2

                displacement = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                rotation_diff = abs(a2 - a1)

                if (
                    displacement > translation_threshold
                    or rotation_diff > rotation_threshold
                ):
                    movements.append((i, j, displacement, rotation_diff))

    return movements


if __name__ == "__main__":
    boxes_image1 = [
        (100, 100, 50, 50, 30),
        (200, 200, 50, 50, 45),
    ]  # (x, y, width, height, angle)
    boxes_image2 = [
        (110, 105, 50, 50, 35),
        (210, 195, 50, 50, 50),
    ]  # Slight movement and rotation

    movements = detect_movement_rotated(boxes_image1, boxes_image2)

    if movements:
        for i, j, displacement, rotation_diff in movements:
            print(
                f"Object {i} in image 1 moved to position {j} in image 2 with displacement {displacement:.2f} and rotation change {rotation_diff:.2f} degrees"
            )
    else:
        print("No significant movements detected.")
