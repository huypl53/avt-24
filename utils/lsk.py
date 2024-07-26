import cv2
import numpy as np
import torch


def image_rotate_without_crop(mat, angle):
    # https://stackoverflow.com/questions/22041699/rotate-an-image-without-cropping-in-opencv-in-c
    # angle in degrees

    height, width = mat.shape[:2]
    image_center = (width / 2, height / 2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)

    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))

    return rotated_mat


def crop_rectangle(image, rect):
    # rect has to be upright

    num_rows = image.shape[0]
    num_cols = image.shape[1]

    if not inside_rect(rect=rect, num_cols=num_cols, num_rows=num_rows):
        print("Proposed rectangle is not fully in the image.")
        return None

    rect_center = rect[0]
    rect_center_x = rect_center[0]
    rect_center_y = rect_center[1]
    rect_width = rect[1][0]
    rect_height = rect[1][1]

    return image[
        rect_center_y
        - rect_height // 2 : rect_center_y
        + rect_height
        - rect_height // 2,
        rect_center_x - rect_width // 2 : rect_center_x + rect_width - rect_width // 2,
    ]


def rect_bbx(rect):
    # Rectangle bounding box for rotated rectangle
    # Example:
    # rotated rectangle: height 4, width 4, center (10, 10), angle 45 degree
    # bounding box for this rotated rectangle, height 4*sqrt(2), width 4*sqrt(2), center (10, 10), angle 0 degree

    box = cv2.boxPoints(rect)

    x_max = int(np.max(box[:, 0]))
    x_min = int(np.min(box[:, 0]))
    y_max = int(np.max(box[:, 1]))
    y_min = int(np.min(box[:, 1]))

    center = (int((x_min + x_max) // 2), int((y_min + y_max) // 2))
    width = int(x_max - x_min)
    height = int(y_max - y_min)
    angle = 0

    return (center, (width, height), angle)


def inside_rect(rect, num_cols, num_rows):
    # Determine if the four corners of the rectangle are inside the rectangle with width and height
    # rect tuple
    # center (x,y), (width, height), angle of rotation (to the row)
    # center  The rectangle mass center.
    # center tuple (x, y): x is regarding to the width (number of columns) of the image, y is regarding to the height (number of rows) of the image.
    # size    Width and height of the rectangle.
    # angle   The rotation angle in a clockwise direction. When the angle is 0, 90, 180, 270 etc., the rectangle becomes an up-right rectangle.
    # Return:
    # True: if the rotated sub rectangle is side the up-right rectange
    # False: else

    rect_center = rect[0]
    rect_center_x = rect_center[0]
    rect_center_y = rect_center[1]

    rect_width, rect_height = rect[1]

    rect_angle = rect[2]

    if (rect_center_x < 0) or (rect_center_x > num_cols):
        return False
    if (rect_center_y < 0) or (rect_center_y > num_rows):
        return False

    # https://docs.opencv.org/3.0-beta/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html
    box = cv2.boxPoints(rect)

    x_max = int(np.max(box[:, 0]))
    x_min = int(np.min(box[:, 0]))
    y_max = int(np.max(box[:, 1]))
    y_min = int(np.min(box[:, 1]))

    if (x_max <= num_cols) and (x_min >= 0) and (y_max <= num_rows) and (y_min >= 0):
        return True
    else:
        return False


def crop_rotated_rectangle(image, rect):
    # Crop a rotated rectangle from a image

    num_rows = image.shape[0]
    num_cols = image.shape[1]

    if not inside_rect(rect=rect, num_cols=num_cols, num_rows=num_rows):
        print("Proposed rectangle is not fully in the image.")
        return None

    rotated_angle = rect[2]

    rect_bbx_upright = rect_bbx(rect=rect)
    rect_bbx_upright_image = crop_rectangle(image=image, rect=rect_bbx_upright)

    rotated_rect_bbx_upright_image = image_rotate_without_crop(
        mat=rect_bbx_upright_image, angle=rotated_angle
    )

    rect_width = rect[1][0]
    rect_height = rect[1][1]

    crop_center = (
        rotated_rect_bbx_upright_image.shape[1] // 2,
        rotated_rect_bbx_upright_image.shape[0] // 2,
    )

    return rotated_rect_bbx_upright_image[
        crop_center[1]
        - rect_height // 2 : crop_center[1]
        + (rect_height - rect_height // 2),
        crop_center[0]
        - rect_width // 2 : crop_center[0]
        + (rect_width - rect_width // 2),
    ]


def xywhr2xyxyxyxy(x):
    """
    Convert batched Oriented Bounding Boxes (OBB) from [xywh, rotation] to [xy1, xy2, xy3, xy4]. Rotation values should
    be in degrees from 0 to 90.

    Args:
        x (numpy.ndarray | torch.Tensor): Boxes in [cx, cy, w, h, rotation] format of shape (n, 5) or (b, n, 5).

    Returns:
        (numpy.ndarray | torch.Tensor): Converted corner points of shape (n, 4, 2) or (b, n, 4, 2).
    """
    cos, sin, cat, stack = (
        (torch.cos, torch.sin, torch.cat, torch.stack)
        if isinstance(x, torch.Tensor)
        else (np.cos, np.sin, np.concatenate, np.stack)
    )

    ctr = x[..., :2]
    w, h, angle = (x[..., i : i + 1] for i in range(2, 5))
    cos_value, sin_value = cos(angle), sin(angle)
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1 = cat(vec1, -1)
    vec2 = cat(vec2, -1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return stack([pt1, pt2, pt3, pt4], -2)


def downsample_image(img: np.ndarray, rate_x: float, rate_y: float):
    assert rate_x < 1 and rate_y < 1
    return cv2.resize(
        img, (0, 0), fx=rate_x, fy=rate_y, interpolation=cv2.INTER_NEAREST
    )
