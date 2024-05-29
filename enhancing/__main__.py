import sys

import cv2

from enhancing.core import adjust_gamma, hist_equalize


def main():
    try:

        im_path = sys.argv[1]
        result_path = sys.argv[2]

        im = cv2.imread(im_path)
        # h, w = im.shape[:2]
        # n_h, n_w = int(1024 / w * h), 1024

        # im_small = cv2.resize(im, (n_w, n_h))
        # cv2.imshow("origin", im_small)

        enhanced_im = adjust_gamma(im, 0.4)
        enhanced_im = hist_equalize(im)
        # enhanced_im_small = cv2.resize(enhanced_im, (n_w, n_h))
        cv2.imwrite(result_path, enhanced_im)
    except Exception:
        print("run `bash enhance <path/to/image> <path/to/result/image>`")


if __name__ == "__main__":
    main()
