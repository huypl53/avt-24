from typing import List
import cv2
import numpy as np


def find_boundary_keypoints(binary_image, replicate=True) -> List[np.ndarray]:
    """
    return keypoints: List[List[Tuple[int, int]]] in (x, y) format
    """
    binary_image = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)[1]

    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    ordered_keypoints = []

    for contour in contours:
        if len(contour) > 2:
            if cv2.contourArea(contour) < 0:
                contour = contour[::-1]
            if replicate:
                contour = [*contour, contour[0]]
            ordered_keypoints.append(contour.squeeze())

    return ordered_keypoints
