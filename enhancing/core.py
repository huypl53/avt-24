import cv2
import numpy as np


def adjust_gamma(src_img: np.ndarray, gamma: float) -> np.ndarray:

    assert gamma > 0 and gamma < 1
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

    res = cv2.LUT(src_img, lookUpTable)
    return res


def hist_equalize(
    src_img: np.ndarray, mode: str = "tiles", tileGridSize=(8, 8)
) -> np.ndarray:
    """
    hist_equalize colored image
    """

    assert mode in ["global", "tiles"]
    ycrcb_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2YCrCb)
    ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
    if mode == "global":
        ycrcb_img = cv2.equalizeHist(ycrcb_img)
    if mode == "tiles":
        assert type(tileGridSize) is tuple
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=tileGridSize)
        for i in range(ycrcb_img.shape[-1]):
            ycrcb_img[..., i] = clahe.apply(ycrcb_img[..., i])
    equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)

    return equalized_img
