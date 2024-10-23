import torch
from typing import List, Tuple
import numpy as np
import cv2
import matplotlib.pyplot as plt


def get_im_fft(img, take_half: bool = True):
    if type(img) != torch.Tensor:
        img = torch.from_numpy(img)
    fft_im = torch.fft.rfft2(img, norm="forward").abs()

    if take_half:
        fft_im = fft_im[:, : fft_im.shape[0] // 2]
    else:
        fft_im = fft_im

    if fft_im.requires_grad:
        fft_im = fft_im.detach()
    if fft_im.is_cuda:
        fft_im = fft_im.cpu()
    return fft_im.numpy()


def sub_fft(im1, im2):
    return np.abs(im1 - im2)


def gen_fft_diff_mask(im1, im2, window_size: int = 1024, window_stride: int = 1024):
    """
    Make sure that both images have the same size and lat/lon location
    Stride should be smaller than window size
    """

    mask = []
    i, j = 0, 0

    while i + window_size < im1.shape[0]:
        j = 0
        mask.append([])
        while j + window_size < im1.shape[1]:
            win1 = im1[i : i + window_size, j : j + window_size, ...]
            win2 = im2[i : i + window_size, j : j + window_size, ...]

            fft_diff = sub_fft(get_im_fft(win1), get_im_fft(win2))
            mask[-1].append(np.mean(fft_diff))
            j += window_stride
        win1 = im1[i : i + window_size, im1.shape[1] - window_size : im1.shape[1], ...]
        win2 = im2[i : i + window_size, im2.shape[1] - window_size : im2.shape[1], ...]

        fft_diff = sub_fft(get_im_fft(win1), get_im_fft(win2))
        mask[-1].append(np.mean(fft_diff))
        i += window_stride

    j = 0
    mask.append([])
    while j + window_size < im1.shape[1]:
        win1 = im1[im1.shape[0] - window_size : im1.shape[0], j : j + window_size, ...]
        win2 = im2[im2.shape[0] - window_size : im2.shape[0], j : j + window_size, ...]

        fft_diff = sub_fft(get_im_fft(win1), get_im_fft(win2))
        mask[-1].append(np.mean(fft_diff))
        j += window_stride
    win1 = im1[
        im1.shape[0] - window_size : im1.shape[0],
        im1.shape[1] - window_size : im1.shape[1],
        ...,
    ]
    win2 = im2[
        im2.shape[0] - window_size : im2.shape[0],
        im2.shape[1] - window_size : im2.shape[1],
        ...,
    ]

    fft_diff = sub_fft(get_im_fft(win1), get_im_fft(win2))
    mask[-1].append(np.mean(fft_diff))

    return np.array(mask)


def mask2image(matrix: np.ndarray):
    # Normalize the matrix to be between 0 and 255
    min_val = np.min(matrix)
    max_val = np.max(matrix)

    normalized_matrix = ((matrix - min_val) / (max_val - min_val)) * 255
    normalized_matrix = normalized_matrix.astype(np.uint8)

    return normalized_matrix
