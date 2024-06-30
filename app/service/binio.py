import io
from typing import Callable, Union

import cv2
import numpy as np

from app.connection import ftpTransfer
from log import logger


def read_np_image(read_func: Callable) -> Union[None, np.ndarray]:
    r = io.BytesIO()
    result = read_func(r)
    if not result:
        return None
    image = np.asarray(bytearray(r.getvalue()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


def write_np_image(image: np.ndarray, extension: str, write_func: Callable) -> bool:
    valid_exts = [".jpg", ".png", ".tiff", ".tif"]
    assert extension in valid_exts, f"Image extension must be in {valid_exts}"
    retval, buffer = cv2.imencode(extension, image)
    if not retval:
        return False

    r = io.BytesIO(buffer)
    result = write_func(r)
    return result


def read_ftp_np_image(file_path: str):
    c = lambda f: ftpTransfer.download_file(file_path, f)
    return read_np_image(c)


def read_ftp_bin_image(file_path: str):
    r = io.BytesIO()
    ftpTransfer.download_file(file_path, r)
    return r.getvalue()


def write_ftp_image(image: np.ndarray, extension: str, file_path: str):
    c = lambda f: ftpTransfer.upload_file(file_path, f)
    return write_np_image(image, extension, c)
