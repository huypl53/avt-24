import argparse
import asyncio
import os

from sqlalchemy import select

from app.db.connector import get_db
from app.model.task import TaskMd
from app.service.binio import read_ftp_image, write_ftp_image
from enhancing.core import adjust_gamma, hist_equalize
from log import logger


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_path", type=str, help="Path to image in FTP server")
    parser.add_argument(
        "out_dir",
        type=str,
        default="/data/avt-enhance-result/",
        help="Directory to save image in FTP server",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.4, help="Gamma for brightness enhancement"
    )
    args = parser.parse_args()
    return args


def cli_main():
    try:

        args = parse()
        im_path = args.in_path
        out_dir = args.out_dir
        gamma = args.gamma

        # im = cv2.imread(im_path)
        im = read_ftp_image(im_path)
        if im is None:
            logger.warning(f"Read {im_path} failed!")
            return
        logger.info(f"{im_path} shape: {im.shape}")
        im_name = os.path.basename(im_path)
        bname, extension = im_name.rsplit(".", 1)
        extension = "." + extension
        # h, w = im.shape[:2]
        # n_h, n_w = int(1024 / w * h), 1024

        # im_small = cv2.resize(im, (n_w, n_h))
        # cv2.imshow("origin", im_small)

        enhanced_im = adjust_gamma(im, gamma)
        enhanced_im = hist_equalize(im)
        logger.info(f"Enhance image done!")
        # enhanced_im_small = cv2.resize(enhanced_im, (n_w, n_h))
        # cv2.imwrite(result_path, enhanced_im)
        result_im_name = f"{bname}_result{extension}"
        result_path = os.path.join(out_dir, result_im_name)
        logger.info(f"Write image to {result_path}")
        write_ftp_image(enhanced_im, extension, result_path)
    except Exception as e:
        logger.error(e)
        print("run `bash enhance --help`")


async def async_main():
    a_session = anext(get_db())
    session = await (a_session)
    stmt = (
        select(TaskMd)
        # .where(TaskMd.task_type == 4)
        .where(TaskMd.task_stat < 0).order_by(TaskMd.task_stat.desc())
    )
    tasks = await session.execute(stmt)
    print(f"Tasks {list(tasks)}")
    import pdb

    pdb.set_trace()
    print(tasks.scalars())
    for e in tasks:
        print(e)
    for i, t in enumerate(list(tasks)):
        print(i, dict(t))

    # await asyncio.sleep(3)
    await session.close()


if __name__ == "__main__":
    cli_main()
