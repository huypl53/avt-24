import json
import os
import subprocess
from typing import List, Tuple

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


def pixel_point_to_lat_long(
    im_path: str, points: List[Tuple[float, float]] | np.ndarray
) -> List[Tuple[float, float]]:

    """
    points: N points in format of (x, y)
    """
    # 'size': [9982, 5484],  width, height
    r = subprocess.run(["gdalinfo", im_path, "-json"], capture_output=True, text=True)
    result = json.loads(r.stdout)
    corners = result["cornerCoordinates"]
    tl, bl, br, tr = (
        corners["upperLeft"],
        corners["lowerLeft"],
        corners["lowerRight"],
        corners["upperRight"],
    )
    w, h, _ = result["size"]

    dxy = [(p[0] / w, p[1] / h) for p in points]
    ll_points: List[Tuple[float, float]] = []
    for d in dxy:
        dx, dy = d

        lat_top = tl[0] + (tr[0] - tl[0]) * dx
        lat_bot = bl[0] + (br[0] - bl[0]) * dx
        lat = lat_top + (lat_bot - lat_top) * dy

        lon_top = tl[1] + (tr[1] - tl[1]) * dx
        lon_bot = bl[1] + (br[1] - bl[1]) * dx
        lon = lon_top + (lon_bot - lon_top) * dy

        ll_points.append((lat, lon))

    return ll_points
