import matplotlib.pyplot as plt
import numpy as np
import pytest

from tests.config import get_sample_photo
from utils.lsk import downsample_image
from utils.plot import plot_images


@pytest.mark.parametrize("image, down_x, down_y", [get_sample_photo(), 0.6, 0.6])
def test_downsample_image(img: np.ndarray, rate_x: float, rate_y: float):
    downscaled_image = downsample_image(img, rate_x, rate_y)
    result = plot_images([img, downscaled_image, 2])
    plt.imshow(result)
    pass


if __name__ == "__main__":
    pass
