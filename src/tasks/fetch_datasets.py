import argparse
import os
import subprocess
from logging import DEBUG
from pathlib import Path
from typing import List, Optional

from src.core.logger import get_logger

logger = get_logger(DEBUG)


def run_shell(commands: List[str]):
    try:
        out = subprocess.run(commands, check=True, capture_output=True)
        print(out.stdout)
    except subprocess.CalledProcessError as e:
        # TODO: shellのエラー表示してプロセスを終了するようにする
        print(e.stderr)
        raise


def fetch_datasets(compe_name: str, workdir_path: Path, rm_zip: bool = False) -> None:
    """download datasets

    Args:
        compe_name:
        zipfile:
        workdir_path:

    NOTE:
        this function needs shell commands. I checked this function was working on Ubuntu20.04

        you have to install commands below

        * kaggle api
        * unzip

        and kaggle-api needs kaggle.json, so you must put kaggle.json under $HOME/.kaggle

        Ref:
        [1] kaggle-api (https://github.com/Kaggle/kaggle-api)
    """
    data_dir = workdir_path / "input"
    data_dir.mkdir(exist_ok=True)
    logger.info(f" #### data_dir: {data_dir} #### ")
    zipfile = str(data_dir / (compe_name + ".zip"))
    dst_dir = str(data_dir / compe_name) + "/"

    logger.info(" #### start to download. #### ")
    run_shell(
        [
            "kaggle",
            "competitions",
            "download",
            "-c",
            compe_name,
            "-p",
            str(data_dir),
        ],
    )
    logger.info(" #### completed download. #### ")

    logger.info(" #### start to unzip. #### ")
    # assume that zip file name is same as compe name
    run_shell(
        [
            "unzip",
            "-u",
            zipfile,
            "-d",
            dst_dir,
        ]
    )

    if rm_zip:
        run_shell(["rm", zipfile])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compe_name", type=str)
    parser.add_argument("--work_dir", default=None, type=Optional[str])
    parser.add_argument("--rm_zip", action="store_true")
    args = parser.parse_args()
    compe_name = args.compe_name
    work_dir: Optional[str] = args.work_dir

    if work_dir is None:
        logger.info(" workd_dir is not specified... ")
        work_dir = (
            os.getenv(
                "WORK_DIR",
            )
            or Path(__file__).resolve().parents[2]
        )
    logger.info(f" #### work_dir : {work_dir} #### ")

    if args.rm_zip:
        fetch_datasets(compe_name, work_dir, rm_zip=True)
    else:
        fetch_datasets(compe_name, work_dir)


if __name__ == "__main__":
    main()

