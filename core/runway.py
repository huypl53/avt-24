from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as cv2
import numpy as np

from log import logger


@dataclass
class Runway:
    bbox: Tuple  # Output of cv2.minAreaRect
    start_point: Tuple[float, float]  # (lat, lon)
    end_point: Tuple[float, float]  # (lat, lon)
    length_meters: float
    width_meters: float  # Added width field
    center_point: Tuple[float, float]  # Added center point for reference
    angle: Optional[float] = None

    def __str__(self):
        return f"Runway: {self.length_meters:.1f}m x {self.width_meters:.1f}m, at {self.center_point}"


def split_image(
    image: np.ndarray, patch_size: int = 2048, overlap: int = 200
) -> List[Tuple]:
    """
    Split large image into overlapping patches.
    Returns list of (patch, (x_offset, y_offset)) tuples.
    """
    # img = cv2.imread(image)
    img = image
    height, width = img.shape[:2]
    patches = []

    for y in range(0, height, patch_size - overlap):
        for x in range(0, width, patch_size - overlap):
            end_y = min(y + patch_size, height)
            end_x = min(x + patch_size, width)
            # Ensure patch has minimum size
            if end_y - y < patch_size // 4 or end_x - x < patch_size // 4:
                continue
            patch = img[y:end_y, x:end_x]
            patches.append((patch, (x, y)))

    return patches


def merge_boxes(
    boxes: List[Tuple],
    offsets: List[Tuple],
    min_angle_diff: float = 15.0,
    max_distance: float = 100,
) -> List[Tuple]:
    """
    Merge runway boxes that are likely part of the same runway.

    Args:
        boxes: List of bounding boxes
        offsets: List of patch offsets
        min_angle_diff: Minimum angle difference to consider boxes aligned
        max_distance: Maximum distance between boxes to be merged

    Returns:
        List of merged bounding boxes

    Raises:
        ValueError: If input parameters are invalid
    """
    try:
        # Validate inputs
        if len(boxes) != len(offsets):
            raise ValueError("Number of boxes and offsets must match")

        if not boxes:
            logger.warning("No boxes to merge")
            return []

        if min_angle_diff < 0 or min_angle_diff > 180:
            raise ValueError("min_angle_diff must be between 0 and 180")

        if max_distance < 0:
            raise ValueError("max_distance must be positive")

        # Adjust boxes based on patch offsets
        adjusted_boxes = []
        for box, (offset_x, offset_y) in zip(boxes, offsets):
            if not isinstance(box, tuple) or len(box) != 3:
                raise ValueError(f"Invalid box format: {box}")

            center, (w, h), angle = box
            adjusted_center = (center[0] + offset_x, center[1] + offset_y)
            adjusted_boxes.append((adjusted_center, (w, h), angle))

        # Merge similar boxes
        merged = []
        used = set()

        for i, box1 in enumerate(adjusted_boxes):
            if i in used:
                continue

            current_group = [box1]
            used.add(i)

            for j, box2 in enumerate(adjusted_boxes[i + 1 :], i + 1):
                if j in used:
                    continue

                # Check if boxes are aligned (similar angle)
                angle_diff = abs(box1[2] - box2[2]) % 180
                angle_diff = min(angle_diff, 180 - angle_diff)

                if angle_diff > min_angle_diff:
                    continue

                # Check if boxes are close enough
                dist = np.sqrt(
                    (box1[0][0] - box2[0][0]) ** 2 + (box1[0][1] - box2[0][1]) ** 2
                )

                if dist > max_distance:
                    continue

                current_group.append(box2)
                used.add(j)

            # Merge boxes in current group
            if len(current_group) == 1:
                merged.append(current_group[0])
            else:
                # Calculate average center, dimensions, and angle
                centers = np.array([box[0] for box in current_group])
                merged_center = tuple(centers.mean(axis=0))

                dims = np.array([box[1] for box in current_group])
                merged_dims = tuple(dims.max(axis=0))

                angles = np.array([box[2] for box in current_group])
                merged_angle = float(np.median(angles))

                merged.append((merged_center, merged_dims, merged_angle))

        logger.info(f"Merged {len(boxes)} boxes into {len(merged)} boxes")
        return merged

    except Exception as e:
        raise ValueError(f"Error merging boxes: {str(e)}")


def process_runway_image(
    image: np.ndarray,
    inference_segmentor,
    pixel_to_latlon,
    calculate_distance,
    min_runway_length: float = 500.0,
    patch_size: int = 2048,
    overlap: int = 200,
) -> List[Runway]:
    """
    Main function to process large image and detect runways.

    Args:
        image: Numpy array representing the image
        inference_segmentor: Function that takes image and returns bounding boxes
        pixel_to_latlon: Function that converts pixel coordinates to lat/lon
        calculate_distance: Function that calculates distance between lat/lon points
        min_runway_length: Minimum runway length in meters
        patch_size: Size of image patches
        overlap: Overlap between patches

    Returns:
        List of Runway objects containing filtered runway information
    """
    # Split image into patches
    patches = split_image(image, patch_size, overlap)

    # Process each patch
    all_boxes = []
    all_offsets = []

    for i, (patch, offset) in enumerate(patches):
        try:
            boxes = inference_segmentor(patch)
            if not isinstance(boxes, (list, tuple)):
                raise TypeError(
                    f"inference_segmentor returned {type(boxes)}, expected list"
                )

            all_boxes.extend(boxes)
            all_offsets.extend([offset] * len(boxes))

            logger.info(
                f"Processed patch {i+1}/{len(patches)}: found {len(boxes)} runways"
            )

        except Exception as e:
            logger.error(f"Error processing patch {i}: {str(e)}")
            continue

    if not all_boxes:
        logger.warning("No runways detected in any patch")
        return []

    # Merge boxes from different patches
    merged_boxes = merge_boxes(all_boxes, all_offsets)

    # Convert to Runway objects and filter by length
    runways = []
    for box in merged_boxes:
        try:
            center, (width, height), angle = box

            # Calculate runway endpoints for length
            dx = width * np.cos(np.radians(angle)) / 2
            dy = width * np.sin(np.radians(angle)) / 2

            start_pixel = (center[0] - dx, center[1] - dy)
            end_pixel = (center[0] + dx, center[1] + dy)

            # Calculate width points
            wx = height * np.cos(np.radians(angle + 90)) / 2
            wy = height * np.sin(np.radians(angle + 90)) / 2

            width_point1 = (center[0] - wx, center[1] - wy)
            width_point2 = (center[0] + wx, center[1] + wy)

            # Convert all points to lat/lon
            start_latlon = pixel_to_latlon(start_pixel)
            end_latlon = pixel_to_latlon(end_pixel)
            center_latlon = pixel_to_latlon(center)
            width_latlon1 = pixel_to_latlon(width_point1)
            width_latlon2 = pixel_to_latlon(width_point2)

            # Validate all coordinates
            for lat, lon in [
                start_latlon,
                end_latlon,
                center_latlon,
                width_latlon1,
                width_latlon2,
            ]:
                validate_coordinates(lat, lon)

            # Calculate length and width
            length = calculate_distance(start_latlon, end_latlon)
            width = calculate_distance(width_latlon1, width_latlon2)

            # Validate measurements
            if not isinstance(length, (int, float)) or not isinstance(
                width, (int, float)
            ):
                raise TypeError(
                    f"Invalid distance type: length={type(length)}, width={type(width)}"
                )

            if length >= min_runway_length:
                runways.append(
                    Runway(
                        bbox=box,
                        start_point=start_latlon,
                        end_point=end_latlon,
                        length_meters=length,
                        width_meters=width,
                        center_point=center_latlon,
                        angle=angle,
                    )
                )

                logger.info(f"Found runway: length={length:.1f}m, width={width:.1f}m")

        except Exception as e:
            logger.error(f"Error processing runway: {str(e)}")
            continue

    logger.info(f"Found {len(runways)} runways of minimum length {min_runway_length}m")
    return runways


def validate_coordinates(lat: float, lon: float) -> bool:
    """Validate latitude and longitude values"""
    if not (-90 <= lat <= 90):
        raise ValueError(f"Invalid latitude: {lat}")
    if not (-180 <= lon <= 180):
        raise ValueError(f"Invalid longitude: {lon}")
    return True
