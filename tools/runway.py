import numpy as np
from typing import Tuple, List
import os
from pyproj import Geod
import sys
import traceback
import json
import random
import numpy as np
from datetime import datetime
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.type import ObjectClass
from tools.gis import download_arcgis


def calculate_bounding_box_wgs84(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    buffer_min: float,
    buffer_max: float,
) -> Tuple[float, float, float, float]:
    """
    Calculate a bounding box around a runway with random buffer zone in WGS84.
    Uses pyproj's Geod for accurate geodesic calculations.

    Args:
        lat1, lon1: Coordinates of runway start in WGS84 (EPSG:4326)
        lat2, lon2: Coordinates of runway end in WGS84 (EPSG:4326)
        buffer_min: Minimum buffer distance in meters
        buffer_max: Maximum buffer distance in meters

    Returns:
        Tuple of (min_lat, min_lon, max_lat, max_lon) in WGS84
    """
    # Use WGS84 ellipsoid for calculations
    geod = Geod(ellps="WGS84")

    # Calculate runway azimuth and back azimuth
    try:
        az12, az21, dist = geod.inv(lon1, lat1, lon2, lat2)
    except ValueError:
        # If the points are too close, handle the exception
        az12 = 0
        az21 = 0
        dist = 0

    # Generate random buffer distance
    buffer_meters = random.uniform(buffer_min, buffer_max)

    # Calculate buffer points perpendicular to runway direction
    # For each runway end, calculate points on both sides
    points = []

    # Buffer points at runway start
    try:
        lon_start_left, lat_start_left, _ = geod.fwd(
            lon1, lat1, az12 - 90, buffer_meters
        )
        lon_start_right, lat_start_right, _ = geod.fwd(
            lon1, lat1, az12 + 90, buffer_meters
        )
    except ValueError:
        # If the points are too close to the poles, handle the exception
        lon_start_left, lat_start_left = lon1, lat1
        lon_start_right, lat_start_right = lon1, lat1

    # Buffer points at runway end
    try:
        lon_end_left, lat_end_left, _ = geod.fwd(lon2, lat2, az12 - 90, buffer_meters)
        lon_end_right, lat_end_right, _ = geod.fwd(lon2, lat2, az12 + 90, buffer_meters)
    except ValueError:
        # If the points are too close to the poles, handle the exception
        lon_end_left, lat_end_left = lon2, lat2
        lon_end_right, lat_end_right = lon2, lat2

    # Add diagonal buffer points
    for lon, lat in [(lon1, lat1), (lon2, lat2)]:
        for angle in range(0, 360, 45):  # 8 points around each end
            try:
                lon_buf, lat_buf, _ = geod.fwd(lon, lat, angle, buffer_meters)
                points.append((lat_buf, lon_buf))
            except ValueError:
                # If the points are too close to the poles, handle the exception
                pass

    # Add the buffer corner points
    points.extend(
        [
            (lat_start_left, lon_start_left),
            (lat_start_right, lon_start_right),
            (lat_end_left, lon_end_left),
            (lat_end_right, lon_end_right),
        ]
    )

    # Calculate bounding box
    min_lat = min(p[0] for p in points)
    max_lat = max(p[0] for p in points)
    min_lon = min(p[1] for p in points)
    max_lon = max(p[1] for p in points)

    # Shift the bounding box to be off-center
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2

    # Generate random offsets within 25% of the bounding box size
    lat_offset = random.uniform(-0.25 * (max_lat - min_lat), 0.25 * (max_lat - min_lat))
    lon_offset = random.uniform(-0.25 * (max_lon - min_lon), 0.25 * (max_lon - min_lon))

    min_lat = center_lat - 0.5 * (max_lat - min_lat) + lat_offset
    max_lat = center_lat + 0.5 * (max_lat - min_lat) + lat_offset
    min_lon = center_lon - 0.5 * (max_lon - min_lon) + lon_offset
    max_lon = center_lon + 0.5 * (max_lon - min_lon) + lon_offset

    return min_lat, min_lon, max_lat, max_lon


def read_runway_coordinates(filepath: str, reverse=False) -> List[tuple]:
    """
    Read runway coordinates from text file in WGS84 (EPSG:4326).
    Each line should contain: lat1, lon1, lat2, lon2 in decimal degrees
    """
    runways = []
    with open(filepath, "r") as f:
        for i, line in enumerate(f, 1):
            try:
                coords = [float(x.strip()) for x in line.split(",")]
                if len(coords) < 4:
                    print(
                        f"Line {i}: Skipping invalid line (need 4 coordinates): {line}"
                    )
                    continue
                if reverse:
                    coords = np.array(coords).reshape((-1, 2))[..., ::-1].reshape(-1)

                # Basic validation of WGS84 coordinates
                for j, coord in enumerate(coords[:4]):
                    if j % 2 == 0:  # Longitude
                        if not -180 <= coord <= 180:
                            print(
                                f"Line {i}: Invalid longitude {coord} (must be between -180 and 180)"
                            )
                            break
                    else:  # Latitude
                        if not -90 <= coord <= 90:
                            print(
                                f"Line {i}: Invalid latitude {coord} (must be between -90 and 90)"
                            )
                            break
                else:  # All coordinates valid
                    runways.append(tuple(coords))

            except ValueError:
                print(f"Line {i}: Skipping invalid line (not numbers): {line}")
                continue
    return runways


def download_runway_images(
    runway_file: str,
    output_dir: str,
    resolution: int = 1,
    buffer_min: float = 300,
    buffer_max: float = 700,
    sample_num: int = 1,
    cls=ObjectClass.RUNWAY.name,
):
    """
    Download TIF images for each runway with specified resolution.
    Coordinates are expected in WGS84 (EPSG:4326).

    Args:
        runway_file: Path to text file containing runway coordinates
        output_dir: Directory to save downloaded TIF files
        resolution: Desired resolution in meters per pixel (1 or 3)
        buffer_min: Minimum buffer zone around runway in meters
        buffer_max: Maximum buffer zone around runway in meters
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read runway coordinates
    runways = read_runway_coordinates(runway_file, reverse=True)
    print(f"Found {len(runways)} valid runways")

    # Process each runway
    err_file = os.path.join(output_dir, "error.log")
    err_indices = []
    with open(err_file, "a") as f:
        f.write(runway_file + "\n")

    for i, (lon1, lat1, lon2, lat2, lon3, lat3, lon4, lat4) in tqdm(
        enumerate(runways), leave=False, desc="random buffer"
    ):
        # Calculate bounding box using WGS84-aware function with random buffer
        for _ in range(sample_num):
            min_lat, min_lon, max_lat, max_lon = calculate_bounding_box_wgs84(
                lat1, lon1, lat2, lon2, buffer_min, buffer_max
            )

            t = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
            prefix = f"{cls}_{t}"
            # Generate output filename

            filename = f"{prefix}_{i+1}_{resolution}m.tif"
            output_path = os.path.join(output_dir, filename)

            # Skip if file already exists
            if os.path.exists(output_path):
                print(f"Skipping {filename} - already exists")
                continue

            try:
                # Call your download function (assumes it accepts WGS84 coordinates)
                download_arcgis(
                    [min_lon, min_lat, max_lon, max_lat], output_path=output_path
                )

                # Save runway coordinates and bbox for this image
                coord_file = os.path.join(
                    output_dir, f"{prefix}_{i+1}_{resolution}m.json"
                )
                output = dict(
                    {
                        "meta": "coordinates (WGS84 EPSG:4326)",
                        "coords": [lon1, lat1, lon2, lat2, lon3, lat3, lon4, lat4],
                        "bbox": [min_lat, min_lon, max_lat, max_lon],
                    }
                )
                with open(coord_file, "w") as f:
                    # f.write(f"# Runway coordinates (WGS84 EPSG:4326)\n")
                    # f.write(f"runway_start: {lat1},{lon1}\n")
                    # f.write(f"runway_end: {lat2},{lon2}\n")
                    # f.write(f"bbox: {min_lat},{min_lon},{max_lat},{max_lon}\n")
                    json.dump(output, f)

                print(f"Successfully downloaded {filename}")

            except Exception as e:
                print(f"Error downloading runway {i+1}: {str(e)}")
                err_indices.append(i)
                with open(err_file, "a") as f:
                    f.write(f"{i}: {str(e)}\n")


def main():
    # Configuration
    runway_file = r"sample/runway.txt"  # Your input file with WGS84 coordinates
    output_dir = r"sample/runway"  # Directory to save downloaded images
    resolution = 3  # 1 meter per pixel
    buffer_min = 100  # Minimum buffer around runway in meters
    buffer_max = 10000  # Maximum buffer around runway in meters

    try:
        download_runway_images(
            runway_file=runway_file,
            output_dir=output_dir,
            resolution=resolution,
            buffer_min=buffer_min,
            buffer_max=buffer_max,
        )
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print("An error occurred:")
        # Extracting traceback details line by line
        traceback_details = traceback.extract_tb(exc_traceback)
        for frame in traceback_details:
            print(
                f"File: {frame.filename}, Line: {frame.lineno}, Function: {frame.name}, Code: {frame.line}"
            )
        print(f"Error: {str(e)}")


def crawl(sample_num=10):

    runway_file = r"sample/runway.txt"  # Your input file with WGS84 coordinates
    output_dir = r"sample/runway"  # Directory to save downloaded images
    buffer_min = 100  # Minimum buffer around runway in meters
    buffer_max = 10000  # Maximum buffer around runway in meters

    resolutions = [1, 3]  # meter per pixel
    for resolution in tqdm(resolutions, leave=False, desc="Resolution"):
        try:
            download_runway_images(
                runway_file=runway_file,
                output_dir=output_dir,
                resolution=resolution,
                buffer_min=buffer_min,
                buffer_max=buffer_max,
                sample_num=sample_num,
                cls=ObjectClass.RUNWAY.name,
            )
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print("An error occurred:")
            # Extracting traceback details line by line
            traceback_details = traceback.extract_tb(exc_traceback)
            for frame in traceback_details:
                print(
                    f"File: {frame.filename}, Line: {frame.lineno}, Function: {frame.name}, Code: {frame.line}"
                )
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    # main()
    crawl(3)
