import json
import math
import os
import subprocess
from typing import List, Tuple

import geopy.distance
import numpy as np


# make sure gdal was installed
def read_image_corner_coords(im_path: str) -> List[Tuple[float, float]]:
    """
    return 4 corners' coordinates
    {'upperLeft': [11914294.93, 2388548.401],
     'lowerLeft': [11914294.93, 2383064.107],
     'lowerRight': [11924276.468, 2383064.107],
     'upperRight': [11924276.468, 2388548.401],
     'center': [11919285.699, 2385806.254]}
    """
    assert os.path.isfile(im_path)
    r = subprocess.run(["gdalinfo", im_path, "-json"], capture_output=True, text=True)
    result = json.loads(r.stdout)
    return result["cornerCoordinates"]


def read_tif_meta(im_path: str):
    r = subprocess.run(["gdalinfo", im_path, "-json"], capture_output=True, text=True)
    result = json.loads(r.stdout)
    return result


def pixel_point_to_lat_long(
    points: List[Tuple[float, float]] | np.ndarray, tif_meta
) -> List[List[float]]:
    """convert list of point(x, y) into lat/long
    points: N points in format of (x, y)
    Returns:
        List[ [lat, long] x N]: _description_
    """
    # 'size': [9982, 5484],  width, height
    # check more at https://gdal.org/programs/gdalinfo.html

    # corners = result["cornerCoordinates"]
    # tl, bl, br, tr = (
    #     corners["upperLeft"],
    #     corners["lowerLeft"],
    #     corners["lowerRight"],
    #     corners["upperRight"],
    # )

    # decimal degree
    corners = tif_meta["wgs84Extent"]["coordinates"]
    tl, bl, br, tr = corners[0][:4]
    w, h = tif_meta["size"]

    dxy = [(p[0] / w, p[1] / h) for p in points]
    ll_points: List[List[float]] = []
    for d in dxy:
        dx, dy = d

        lat_top = tl[1] + (tr[1] - tl[1]) * dx
        lat_bot = bl[1] + (br[1] - bl[1]) * dx
        lat = lat_top + (lat_bot - lat_top) * dy

        lon_top = tl[0] + (tr[0] - tl[0]) * dx
        lon_bot = bl[0] + (br[0] - bl[0]) * dx
        lon = lon_top + (lon_bot - lon_top) * dy

        ll_points.append([lat, lon])

    return ll_points


def latlong2meter(lon1, lat1, lon2, lat2):
    return geopy.distance.geodesic((lat1, lon1), (lat2, lon2)).m


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great-circle distance between two points
    on the Earth specified in decimal degrees of latitude and longitude.
    Returns the distance in meters.
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    r = 6371000  # Radius of Earth in meters. Use 6371 for kilometers
    distance_m = r * c

    return distance_m


if __name__ == "__main__":
    # Example usage:
    lon1, lat1 = -73.935242, 40.730610  # New York City coordinates
    lon2, lat2 = -0.127758, 51.507351  # London coordinates

    distance_m = haversine(lon1, lat1, lon2, lat2)
    distance_km = distance_m / 1000

    print(f"Distance: {distance_m:.2f} meters")
    print(f"Distance: {distance_km:.2f} kilometers")
