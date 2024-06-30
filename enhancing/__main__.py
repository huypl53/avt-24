import argparse
import asyncio
import json
import os
from typing import List, Tuple

from sqlalchemy import select

from app.connection import ftpTransfer
from app.db.connector import get_db
from app.model.task import TaskMd
from app.schema import EnhancementOutput, EnhancementParam
from app.service.binio import read_ftp_np_image, write_ftp_image
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


def process_image(im_path: str, out_dir: str, gamma=0.4) -> Tuple[bool, str]:
    try:
        # im = cv2.imread(im_path)
        im = read_ftp_np_image(im_path)
        if im is None:
            msg = f"Read {im_path} failed!"
            logger.warning(msg)
            return False, msg
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
        return True, result_path
    except Exception as e:
        logger.error(e)
        print("run `bash enhance --help`")
        return False, str(e)


def cli_main():
    try:
        args = parse()
        im_path = args.in_path
        out_dir = args.out_dir
        gamma = args.gamma
        process_image(im_path, out_dir, gamma)
    except Exception as e:
        logger.error(e)
        print("run `bash enhance --help`")


async def async_main():
    while True:
        a_session = anext(get_db())
        session = await (a_session)
        stmt = (
            select(TaskMd)
            .where(TaskMd.type == 4)
            .where(TaskMd.task_stat < 0)
            .order_by(TaskMd.task_stat.desc())
        )
        results = await session.execute(stmt)
        mapping_results = results.mappings().all()
        tasks: List[TaskMd] = [m["TaskMd"] for m in mapping_results]

        print("----------")
        try:
            # HACK: update selected rows
            for i, t in enumerate(tasks):
                if i == 1:
                    break  # update only one
                param_dict = json.loads(t.task_param)
                params = EnhancementParam(**param_dict)
                ftpTransfer.mkdir(params.out_dir)
                status, result = process_image(
                    params.input_file, params.out_dir, params.gamma
                )

                if not status:
                    t.task_stat = 0  # task got error
                    t.task_message = result
                    continue
                output = EnhancementOutput(output_file=result)

                t.task_param = json.dumps(params.model_dump())
                t.task_output = json.dumps(output.model_dump())
                t.process_id = os.getpid()
                t.task_stat = 1  # task finished
                t.task_message = "successful"
        except Exception as e:
            logger.error(e)

        print("----------")
        await session.commit()
        await session.close()
        await asyncio.sleep(3)


def row2dict(row):
    d = {}
    for column in row.__table__.columns:
        d[column.name] = str(getattr(row, column.name))

    return d


if __name__ == "__main__":
    cli_main()
