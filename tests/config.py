import urllib

import cv2
import numpy as np


def get_sample_photo(
    url: str = "https://images.pexels.com/photos/2907679/pexels-photo-2907679.jpeg",
) -> np.ndarray:
    req = urllib.request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)  # 'Load it as it is'
    return img
