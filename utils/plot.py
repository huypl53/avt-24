from io import BytesIO

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


def plt_to_cv_image(fig) -> np.ndarray:
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(img_arr, 1)
    buf.close()
    return img


def plot_images(ncols: int, images: np.ndarray) -> np.ndarray:
    n_images = len(images)
    nrows = n_images // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(24, 16))
    for i in range(n_images):
        im = images[i]
        r = (i + 1) // ncols
        c = i % ncols
        axe: Axes = axes[r, c]
        axe.imshow(im)
        axe.axis("off")
        plt.tight_layout()
    cv_im = plt_to_cv_image(fig)
    return cv_im
