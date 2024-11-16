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
                contour = np.array([*contour, contour[0]])
            ordered_keypoints.append(contour.squeeze())

    return ordered_keypoints


def get_rotated_bbox_corners(bbox):
    """
    Get the coordinates of the four corners of a rotated bounding box.

    Parameters:
    bbox (tuple): Rotated bounding box in the format ((x, y), (width, height), angle).

    Returns:
    tuple: Coordinates of the four corners of the rotated bounding box in the format (x1, y1, x2, y2, x3, y3, x4, y4).
    """
    (x, y), (width, height), angle = bbox

    # Calculate the four corners of the rotated bounding box
    center = np.array([x, y])
    vertices = np.array(
        [
            [-width / 2, -height / 2],
            [-width / 2, height / 2],
            [width / 2, height / 2],
            [width / 2, -height / 2],
        ]
    )

    # Add a column of ones to make homogeneous coordinates
    vertices_homogeneous = np.hstack([vertices, np.ones((4, 1))])

    # Get rotation matrix and apply transformation
    R = cv2.getRotationMatrix2D((0, 0), angle, 1)
    rotated_vertices = vertices_homogeneous @ R.T

    # Translate the rotated vertices to the actual center of the bounding box
    corners = rotated_vertices + center

    # Unpack the coordinates of the four corners
    x1, y1 = corners[0]
    x2, y2 = corners[1]
    x3, y3 = corners[2]
    x4, y4 = corners[3]

    return (x1, y1, x2, y2, x3, y3, x4, y4)


def mask2rbboxes(mask_image):
    """
    Detect rotated bounding boxes of mask areas in the given binary mask image.

    Parameters:
    mask_image (numpy.ndarray): Binary mask image with pixel values of 0 or 255.

    Returns:
    list: List of rotated bounding box coordinates in the format ((x, y), (width, height), angle).
    """
    # Find contours in the mask image
    contours, _ = cv2.findContours(
        mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    # Iterate through the contours and get the rotated bounding boxes
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        boxes.append(rect)

    return boxes


if __name__ == "__main__":
    # Example usage
    mask_image = cv2.imread(
        "/workspace/mmsegmentation/demo/smooth.png", cv2.IMREAD_GRAYSCALE
    )

    rotated_bboxes = mask2rbboxes(mask_image)

    rbbox_image = np.zeros_like(mask_image)
    # Print the rotated bounding box coordinates
    for bbox in rotated_bboxes:
        print(bbox)
        cv2.drawContours(
            rbbox_image,
            [np.intp(cv2.boxPoints(bbox))],
            0,
            (255 // 2),
            2,
        )
    cv2.imwrite("/workspace/mmsegmentation/demo/smooth-rbbox.png", rbbox_image)
