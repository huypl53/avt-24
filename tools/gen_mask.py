from enum import Enum
import numpy as np
import rasterio
import cv2
import json
import os


class ObjectClass(Enum):
    RUNWAY = 1
    AIRPORT = 2
    SHIP = 3
    RADOME = 4


def validate_polygon(points, bbox):
    """
    Validate if the polygon points form a valid quadrilateral and match bbox.
    Args:
        points: numpy array of shape (4, 2) containing [lon, lat] pairs
        bbox: [min_lat, min_lon, max_lat, max_lon]
    Returns:
        bool: True if valid, False otherwise
        str: Error message if invalid, empty string if valid
    """
    if len(points) != 4:
        return False, "Polygon must have exactly 4 points"

    # Check if points are unique
    if len(np.unique(points, axis=0)) != 4:
        return False, "Duplicate points found in polygon"

    # Calculate polygon area
    area = cv2.contourArea(points.astype(np.float32))
    if area <= 0:
        return False, "Invalid polygon area (zero or negative)"

    # Validate coordinates against bbox
    lons = points[:, 0]
    lats = points[:, 1]
    min_lat, min_lon, max_lat, max_lon = bbox

    if (
        not min_lon <= lons.min() <= max_lon
        or not min_lon <= lons.max() <= max_lon
        or not min_lat <= lats.min() <= max_lat
        or not min_lat <= lats.max() <= max_lat
    ):
        return False, "Polygon coordinates outside bbox bounds"

    # Validate coordinate ranges (WGS84)
    if not all(-180 <= lon <= 180 for lon in lons):
        return False, "Invalid longitude values (must be between -180 and 180)"
    if not all(-90 <= lat <= 90 for lat in lats):
        return False, "Invalid latitude values (must be between -90 and 90)"

    return True, ""


def reorder_coordinates(points):
    """
    Reorder coordinates to ensure they form a proper polygon (clockwise order).
    Args:
        points: numpy array of shape (4, 2) containing [lon, lat] pairs
    Returns:
        Reordered points in clockwise order
    """
    # Calculate centroid
    center = np.mean(points, axis=0)

    # Calculate angles between centroid and points
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])

    # Sort points by angle to get clockwise order
    sorted_indices = np.argsort(-angles)
    return points[sorted_indices]


def parse_json_annotation(json_path):
    """
    Parse JSON annotation file and extract object information.
    Returns list of tuples (class_name, points, bbox)
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    annotations = []
    obj = data
    # for obj in data:
    # Extract class name from filename before first underscore
    filename = os.path.basename(json_path)
    class_name = filename.split("_")[0].upper()

    # Get coordinates and bbox
    coords = obj["coords"]
    bbox = obj["bbox"]

    # Convert coordinates to points array [[lon1, lat1], [lon2, lat2], ...]
    points = np.array([[coords[i], coords[i + 1]] for i in range(0, 8, 2)])

    annotations.append((class_name, points, bbox))

    return annotations


def generate_mask(tif_path, annotations):
    """Generate mask image from TIF file and annotations."""
    with rasterio.open(tif_path) as src:
        transform = src.transform
        height = src.height
        width = src.width
        mask = np.zeros((height, width), dtype=np.uint8)

        success = False
        for class_name, points, bbox in annotations:
            # Validate polygon
            is_valid, error_msg = validate_polygon(points, bbox)
            if not is_valid:
                print(
                    f"Warning: Invalid polygon in {class_name}: {error_msg}, file: {tif_path}"
                )
                # continue

            # Reorder points to ensure proper polygon
            points = reorder_coordinates(points)

            # Convert geographic coordinates to pixel coordinates
            pixel_points = []
            for lon, lat in points:
                row, col = rasterio.transform.rowcol(transform, lon, lat)
                pixel_points.append([col, row])

            pixel_points = np.array(pixel_points, dtype=np.int32)

            # Get class value from enum
            try:
                class_value = ObjectClass[class_name].value
            except KeyError:
                print(f"Warning: Unknown class {class_name}, skipping...")
                continue

            # Draw filled polygon
            cv2.fillPoly(mask, [pixel_points], class_value)
            success = True

    return mask, success


def save_mask(mask, output_path, tif_path):
    """Save the mask as a new GeoTIFF with same georeference as input."""
    with rasterio.open(tif_path) as src:
        meta = src.meta.copy()
        meta.update({"dtype": "uint8", "count": 1})

        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(mask, 1)


def main():
    bname = "RUNWAY_2024-11-06_13-37-08-596_1_3m"
    tif_path = f"sample/runway/{bname}.tif"
    json_path = f"sample/runway/{bname}.json"
    output_path = f"sample/runway/{bname}.png"

    # Parse JSON annotations
    annotations = parse_json_annotation(json_path)

    # Generate and save mask
    mask = generate_mask(tif_path, annotations)
    save_mask(mask, output_path, tif_path)


# Example usage
if __name__ == "__main__":
    # main()
    in_dir = "sample/runway"
    from pathlib import Path
    from tqdm import tqdm

    json_files = Path(in_dir).glob("*.json")

    missing_files = []
    for json_path in tqdm(json_files, leave=False, desc="Processing JSON files"):
        # Check if corresponding TIF exists
        tif_path = json_path.with_suffix(".tif")
        if not tif_path.exists():

            missing_files.append(str(tif_path))
            continue

        # Parse JSON annotations
        annotations = parse_json_annotation(json_path)

        # Generate and save mask
        mask, success = generate_mask(tif_path, annotations)
        if not success:
            continue
        save_mask(mask, json_path.with_suffix(".png"), tif_path)
