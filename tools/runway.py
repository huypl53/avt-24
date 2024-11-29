import numpy as np
from typing import Tuple, List
import os
from pyproj import Geod
import sys
import traceback
import json
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime
from tqdm import tqdm
import csv
import re
import unicodedata
from pathlib import Path
from multiprocessing import Pool
from functools import partial
import json
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


def parse_runway_data(csv_content):
    """
    Parse runway data from CSV content into a structured array of dictionaries.

    Args:
        csv_content (str): CSV content as a string

    Returns:
        list: Array of dictionaries with keys 'dataset', 'location', and 'coordinates'
    """

    def clean_text(text):
        """Convert text to ASCII, removing special characters."""
        if not text:
            return ""
        # Normalize unicode characters
        text = unicodedata.normalize("NFKD", text)
        # Remove non-ASCII characters
        text = re.sub(r"[^\x00-\x7F]+", "", text)
        # Replace spaces and special characters with underscores
        text = re.sub(r"[^a-zA-Z0-9]+", "_", text)
        # Remove trailing underscores
        text = text.strip("_")
        return text

    def parse_coordinates(coord_strings):
        """Parse coordinate strings into list of float pairs."""
        coordinates = []
        for coord_str in coord_strings:
            if coord_str:
                # Split by comma and convert to float
                lat, lon = map(float, coord_str.split(","))
                coordinates.extend([lon, lat])
        return coordinates

    result = []
    current_dataset = None
    current_location = None
    current_coordinates = []

    # Split the content into lines and create a CSV reader
    lines = csv_content.strip().split("\n")
    reader = csv.DictReader(lines)

    for row in reader:
        dataset = clean_text(row["dataset"])
        location = clean_text(row["location"])

        # Extract coordinates
        coords = [
            row["lat1, lon1"].strip(),
            row["lat2, lon2"].strip(),
            row["lat3, lon3"].strip(),
            row["lat4, lon4"].strip(),
        ]

        # If this is a new location or dataset
        if (dataset or location) and (
            dataset != current_dataset or location != current_location
        ):
            # Save previous location's data if exists
            if current_coordinates:
                result.append(
                    {
                        "dataset": current_dataset or "",
                        "location": current_location or "",
                        "coordinates": current_coordinates,
                    }
                )
                current_coordinates = []

            current_dataset = dataset or current_dataset
            current_location = location

        # Parse and add coordinates for this runway
        if any(coords):
            current_coordinates.append(parse_coordinates(coords))

    # Add the last location
    if current_coordinates:
        result.append(
            {
                "dataset": current_dataset or "",
                "location": current_location or "",
                "coordinates": current_coordinates,
            }
        )

    return result


def process_runway(
    args, output_dir, resolution, buffer_min, buffer_max, cls, sample_num
):
    """
    Worker function to process a single runway.
    Args:
        args: Tuple of (index, (coordinates, metadata))
        output_dir: Directory to save files
        resolution: Desired resolution in meters per pixel
        buffer_min: Minimum buffer zone around runway in meters
        buffer_max: Maximum buffer zone around runway in meters
        cls: Object class name
    """
    i, (coords, meta) = args
    errors = []

    # Unpack coordinates
    lon1, lat1, lon2, lat2, lon3, lat3, lon4, lat4 = coords

    # Generate multiple samples if requested
    for sample_idx in range(sample_num):
        try:
            # Calculate bounding box using WGS84-aware function with random buffer
            min_lat, min_lon, max_lat, max_lon = calculate_bounding_box_wgs84(
                lat1, lon1, lat2, lon2, buffer_min, buffer_max
            )

            # Generate timestamp and prefix
            t = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
            prefix = f"{cls}_{meta['dataset']}_{meta['location']}_{t}"

            # Generate output filename
            filename = f"{prefix}_{i+1}_{resolution}m.tif"
            output_path = os.path.join(output_dir, filename)

            # Skip if file already exists
            if os.path.exists(output_path):
                continue

            # Download the image
            download_arcgis(
                [min_lon, min_lat, max_lon, max_lat], output_path=output_path
            )

            # Save runway coordinates and bbox for this image
            coord_file = os.path.join(output_dir, f"{prefix}_{i+1}_{resolution}m.json")
            output = {
                "meta": "coordinates (WGS84 EPSG:4326)",
                "dataset": meta["dataset"],
                "location": meta["location"],
                "coords": [lon1, lat1, lon2, lat2, lon3, lat3, lon4, lat4],
                "bbox": [min_lat, min_lon, max_lat, max_lon],
            }

            with open(coord_file, "w") as f:
                json.dump(output, f)

        except Exception as e:
            errors.append((json.dumps(meta), str(e)))

    return errors


def download_runway_images(
    runway_file: str,
    output_dir: str,
    resolution: int = 1,
    buffer_min: float = 300,
    buffer_max: float = 700,
    sample_num: int = 1,
    cls=ObjectClass.RUNWAY.name,
    num_processes: int = None,
):
    """
    Download TIF images for each runway with specified resolution using multiple processes.
    Coordinates are expected in WGS84 (EPSG:4326).
    Args:
        runway_file: Path to CSV file containing runway data
        output_dir: Directory to save downloaded TIF files
        resolution: Desired resolution in meters per pixel (1 or 3)
        buffer_min: Minimum buffer zone around runway in meters
        buffer_max: Maximum buffer zone around runway in meters
        sample_num: Number of samples to generate per runway
        cls: Object class name
        num_processes: Number of processes to use (defaults to CPU count)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read runway data using the parser
    with open(runway_file, "r", encoding="utf-8") as f:
        runway_data = parse_runway_data(f.read())

    # Flatten the runway coordinates for processing
    runways = []
    metadata = []  # Store dataset and location info
    for entry in runway_data:
        for coords in entry["coordinates"]:
            runways.append(coords)
            metadata.append(
                {"dataset": entry["dataset"], "location": entry["location"]}
            )

    print(f"Found {len(runways)} valid runways")

    # Create error log file
    err_file = os.path.join(output_dir, "error.log")

    # Prepare arguments for parallel processing
    process_args = list(enumerate(zip(runways, metadata)))

    # Create a partial function with fixed arguments
    worker = partial(
        process_runway,
        output_dir=output_dir,
        resolution=resolution,
        buffer_min=buffer_min,
        buffer_max=buffer_max,
        cls=cls,
        sample_num=sample_num,
    )

    # Create process pool and run jobs
    with Pool(processes=num_processes) as pool:
        # Use imap_unordered for better performance
        results = list(
            tqdm(
                pool.imap_unordered(worker, process_args),
                total=len(process_args),
                desc="Processing runways",
                leave=False,
            )
        )

    # Collect and log all errors
    all_errors = [error for result in results for error in result]
    if all_errors:
        with open(err_file, "a") as f:
            t = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
            f.write(f"{t}\n")
            for idx, error_msg in all_errors:
                f.write(f"{idx}: {error_msg}\n")

    print(f"Completed processing with {len(all_errors)} errors")
    return all_errors


def find_missing_tif_files(folder_path: str) -> List[Tuple[str, dict]]:
    """
    Find JSON files that don't have corresponding TIF files and return their paths and contents.

    Args:
        folder_path: Path to the folder containing JSON and TIF files

    Returns:
        List of tuples containing (json_path, json_content) for files without TIF
    """
    missing_files = []

    # Get all JSON files in the folder
    json_files = Path(folder_path).glob("*.json")

    for json_path in json_files:
        # Check if corresponding TIF exists
        tif_path = json_path.with_suffix(".tif")

        if not tif_path.exists():
            # Read JSON content
            with open(json_path, "r") as f:
                json_content = json.load(f)
            missing_files.append((str(json_path), json_content))

    return missing_files


def download_missing_tifs(folder_path: str, num_processes=None):
    """
    Main function to process folder and download missing TIF files.

    Args:
        folder_path: Path to the folder containing JSON and TIF files
    """
    missing_files = find_missing_tif_files(folder_path)

    process_args = []
    for json_path, json_content in missing_files:
        # Get basename for output path
        output_path = Path(json_path).with_suffix(".tif")

        # Extract bbox from JSON
        # bbox format: [min_lon, min_lat, max_lon, max_lat]
        bbox = json_content["bbox"]
        min_lat, min_lon, max_lat, max_lon = bbox

        process_args.append([[min_lon, min_lat, max_lon, max_lat], str(output_path)])

    # process_args = [(x, x + 1) for x in range(10)]
    # print(process_args)
    # Create process pool and run jobs
    with Pool(processes=num_processes) as pool:
        # Use imap_unordered for better performance
        results = list(
            tqdm(
                # pool.imap_unordered(download_arcgis, process_args),
                pool.starmap(download_arcgis, process_args),
                total=len(process_args),
                desc="Redownload runways",
                leave=False,
            )
        )

    # err_file = os.path.join(folder_path, "error.log")
    # all_errors = [error for result in results for error in result]
    # if all_errors:
    #     with open(err_file, "a") as f:
    #         t = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    #         f.write(f"{t}\n")
    #         for idx, error_msg in all_errors:
    #             f.write(f"{idx}: {error_msg}\n")


def main():
    # Configuration
    runway_file = (
        r"sample/List-of-airstrip.csv"  # Your input file with WGS84 coordinates
    )
    output_dir = r"sample/runway"  # Directory to save downloaded images
    resolution = 3  # 1 meter per pixel
    buffer_min = 300  # Minimum buffer around runway in meters
    buffer_max = 3500  # Maximum buffer around runway in meters

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


def crawl(sample_num=5):

    runway_file = (
        r"sample/List-of-airstrip.csv"  # Your input file with WGS84 coordinates
    )
    output_dir = r"sample/runway"  # Directory to save downloaded images
    buffer_min = 100  # Minimum buffer around runway in meters
    buffer_max = 3000  # Maximum buffer around runway in meters

    resolutions = [1, 3]  # meter per pixel
    download_missing_tifs(output_dir)
    return
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
