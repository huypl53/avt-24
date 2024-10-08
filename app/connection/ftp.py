import ftplib
from typing import BinaryIO

from app.db.setting import settings
from log import logger


class _FtpConnector:
    def __init__(self) -> None:
        self._try_connect()

    def _reset(self):
        self.ftp_server = ftplib.FTP()
        self.ftp_server.encoding = "utf-8"
        self.connect_status = ""
        self.login_status = ""

    def _try_connect(self):
        self.connect(
            settings.FTP_HOSTNAME,
            settings.FTP_PORT,
            settings.FTP_USERNAME,
            settings.FTP_PASSWORD,
        )

    def connect(self, hostname: str, port: int, username: str, password: str):
        self._reset()
        self.connect_status = self.ftp_server.connect(hostname, port)
        logger.info(f"FTP connect  to {hostname}: {self.connect_status}")
        self.login_status = self.ftp_server.login(username, password)
        logger.info(f"FTP login by {username}: {self.login_status}")
        self.ftp_server.cwd(settings.FTP_WORK_DIR)
        logger.info(f"cwd to {settings.FTP_WORK_DIR}")

    def upload_file(self, file_path: str, file: BinaryIO) -> bool:
        try:
            self.ftp_server.storbinary(f"STOR {file_path}", file)
            self.ftp_server.sendcmd(f"SITE CHMOD {settings.FTP_FILE_PERM} {file_path}")
            logger.info(f"Write {file_path} successfully!")
            return True
        except Exception as e:
            logger.error(f"FTP write to {file_path} failed. Error: {e}")
            self._try_connect()
            return False

    def download_file(self, file_path: str, file: BinaryIO) -> bool:
        try:
            self.ftp_server.retrbinary(f"RETR {file_path}", file.write)
            logger.info(f"Download {file_path} successfully!")
            return True
        except Exception as e:
            logger.error(f"FTP download {file_path} failed. Error: {e}")
            self._try_connect()
            return False

    def mkdir(self, dir_path: str) -> bool:
        try:
            if self.ftp_server.mkd(dir_path):
                self.ftp_server.sendcmd(f"SITE CHMOD 777 {dir_path}")
                return True
            return False
        except Exception as e:
            logger.info(
                f"Creating {dir_path} failed! Please restart program... FTP error: {e}. "
            )
            self._try_connect()
            return False

    def cwd(self, path: str) -> bool:
        current_dir = self.ftp_server.pwd()
        try:
            if self.ftp_server.cwd(path):
                return True
            self.ftp_server.cwd(current_dir)
            return False
        except Exception as e:
            logger.error(e)
            self.ftp_server.cwd(current_dir)
            self._try_connect()
            return False

    def is_dir_existed(self, path: str) -> bool:
        current_dir = self.ftp_server.pwd()
        try:
            if self.ftp_server.cwd(path):
                self.ftp_server.cwd(current_dir)
                return True
            self.ftp_server.cwd(current_dir)
            return False
        except Exception as e:
            logger.error(e)
            self.ftp_server.cwd(current_dir)
            self._try_connect()
            return False

    def close(self):
        self.ftp_server.close()


ftpTransfer = _FtpConnector()
