import numpy as np
from typing import Tuple, List
import os
from pyproj import Geod

from tools.gis import download_arcgis


def calculate_bounding_box_wgs84(
    lat1: float, lon1: float, lat2: float, lon2: float, buffer_meters: float = 500
) -> Tuple[float, float, float, float]:
    """
    Calculate a bounding box around a runway with buffer zone in WGS84.
    Uses pyproj's Geod for accurate geodesic calculations.

    Args:
        lat1, lon1: Coordinates of runway start in WGS84 (EPSG:4326)
        lat2, lon2: Coordinates of runway end in WGS84 (EPSG:4326)
        buffer_meters: Buffer distance in meters

    Returns:
        Tuple of (min_lat, min_lon, max_lat, max_lon) in WGS84
    """
    # Use WGS84 ellipsoid for calculations
    geod = Geod(ellps="WGS84")

    # Calculate runway azimuth and back azimuth
    az12, az21, dist = geod.inv(lon1, lat1, lon2, lat2)

    # Calculate buffer points perpendicular to runway direction
    # For each runway end, calculate points on both sides
    points = []

    # Buffer points at runway start
    lon_start_left, lat_start_left, _ = geod.fwd(lon1, lat1, az12 - 90, buffer_meters)
    lon_start_right, lat_start_right, _ = geod.fwd(lon1, lat1, az12 + 90, buffer_meters)

    # Buffer points at runway end
    lon_end_left, lat_end_left, _ = geod.fwd(lon2, lat2, az12 - 90, buffer_meters)
    lon_end_right, lat_end_right, _ = geod.fwd(lon2, lat2, az12 + 90, buffer_meters)

    # Add diagonal buffer points
    for lon, lat in [(lon1, lat1), (lon2, lat2)]:
        for angle in range(0, 360, 45):  # 8 points around each end
            lon_buf, lat_buf, _ = geod.fwd(lon, lat, angle, buffer_meters)
            points.append((lat_buf, lon_buf))

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

    return min_lat, min_lon, max_lat, max_lon


def read_runway_coordinates(filepath: str) -> List[tuple]:
    """
    Read runway coordinates from text file in WGS84 (EPSG:4326).
    Each line should contain: lat1, lon1, lat2, lon2 in decimal degrees
    """
    runways = []
    with open(filepath, "r") as f:
        for i, line in enumerate(f, 1):
            try:
                coords = [float(x.strip()) for x in line.split(",")]
                if len(coords) != 4:
                    print(
                        f"Line {i}: Skipping invalid line (need 4 coordinates): {line}"
                    )
                    continue

                # Basic validation of WGS84 coordinates
                for j, coord in enumerate(coords):
                    if j % 2 == 0:  # Latitude
                        if not -90 <= coord <= 90:
                            print(
                                f"Line {i}: Invalid latitude {coord} (must be between -90 and 90)"
                            )
                            break
                    else:  # Longitude
                        if not -180 <= coord <= 180:
                            print(
                                f"Line {i}: Invalid longitude {coord} (must be between -180 and 180)"
                            )
                            break
                else:  # All coordinates valid
                    runways.append(tuple(coords))

            except ValueError:
                print(f"Line {i}: Skipping invalid line (not numbers): {line}")
                continue
    return runways


def download_runway_images(
    runway_file: str, output_dir: str, resolution: int = 1, buffer_meters: float = 500
):
    """
    Download TIF images for each runway with specified resolution.
    Coordinates are expected in WGS84 (EPSG:4326).

    Args:
        runway_file: Path to text file containing runway coordinates
        output_dir: Directory to save downloaded TIF files
        resolution: Desired resolution in meters per pixel (1 or 3)
        buffer_meters: Buffer zone around runway in meters
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read runway coordinates
    runways = read_runway_coordinates(runway_file)
    print(f"Found {len(runways)} valid runways")

    # Process each runway
    for i, (lat1, lon1, lat2, lon2) in enumerate(runways):
        # Calculate bounding box using WGS84-aware function
        min_lat, min_lon, max_lat, max_lon = calculate_bounding_box_wgs84(
            lat1, lon1, lat2, lon2, buffer_meters
        )

        # Generate output filename
        filename = f"runway_{i+1}_{resolution}m.tif"
        output_path = os.path.join(output_dir, filename)

        # Skip if file already exists
        if os.path.exists(output_path):
            print(f"Skipping {filename} - already exists")
            continue

        try:
            # Call your download function (assumes it accepts WGS84 coordinates)
            download_arcgis(
                min_lat=min_lat,
                min_lon=min_lon,
                max_lat=max_lat,
                max_lon=max_lon,
                output_path=output_path,
            )

            # Save runway coordinates and bbox for this image
            coord_file = os.path.join(output_dir, f"runway_{i+1}_coords.txt")
            with open(coord_file, "w") as f:
                f.write(f"# Runway coordinates (WGS84 EPSG:4326)\n")
                f.write(f"runway_start: {lat1},{lon1}\n")
                f.write(f"runway_end: {lat2},{lon2}\n")
                f.write(f"bbox: {min_lat},{min_lon},{max_lat},{max_lon}\n")

            print(f"Successfully downloaded {filename}")

        except Exception as e:
            print(f"Error downloading runway {i+1}: {str(e)}")


def main():
    # Configuration
    runway_file = r"C:\Users\BTL86\code\tutors\tmp\runways.txt"  # Your input file with WGS84 coordinates
    output_dir = (
        r"C:\Users\BTL86\code\tutors\tmp\runways"  # Directory to save downloaded images
    )
    resolution = 1  # 1 meter per pixel
    buffer_meters = 500  # 500m buffer around runway

    download_runway_images(
        runway_file=runway_file,
        output_dir=output_dir,
        resolution=resolution,
        buffer_meters=buffer_meters,
    )


if __name__ == "__main__":
    main()
