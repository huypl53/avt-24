import argparse
import logging
import os

from enhancing.core import adjust_gamma, hist_equalize
from service import read_ftp_image, write_ftp_image


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_path", type=str, help="Path to image in FTP server")
    parser.add_argument(
        "out_dir",
        type=str,
        default="/data/",
        help="Directory to save image in FTP server",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.4, help="Gamma for brightness enhancement"
    )
    args = parser.parse_args()
    return args


def main():
    try:

        args = parse()
        im_path = args.in_path
        out_dir = args.out_dir
        gamma = args.gamma

        # im = cv2.imread(im_path)
        im = read_ftp_image(im_path)
        if im is None:
            logging.warning(f"Read {im_path} failed!")
            return
        im_name = os.path.basename(im_path)
        bname, extension = im_name.rsplit(".", 1)
        extension = "." + extension
        # h, w = im.shape[:2]
        # n_h, n_w = int(1024 / w * h), 1024

        # im_small = cv2.resize(im, (n_w, n_h))
        # cv2.imshow("origin", im_small)

        enhanced_im = adjust_gamma(im, gamma)
        enhanced_im = hist_equalize(im)
        # enhanced_im_small = cv2.resize(enhanced_im, (n_w, n_h))
        # cv2.imwrite(result_path, enhanced_im)
        result_im_name = f"{bname}_result{extension}"
        result_path = os.path.join(out_dir, result_im_name)
        write_ftp_image(enhanced_im, extension, result_path)
    except Exception:
        print("run `bash enhance --help`")


if __name__ == "__main__":
    main()
