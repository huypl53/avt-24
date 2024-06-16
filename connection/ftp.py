import ftplib
import logging
from typing import BinaryIO

HOSTNAME = "dataprocess.online"
PORT = 18921
USERNAME = "avt"
PASSWORD = "Pl0d9RQYUJCxZPGw6NJUcb8eJ6ZXdNMw"
WORK_DIR = "/data"


class _FtpConnector:
    def __init__(self) -> None:
        self.ftp_server = ftplib.FTP()
        self.ftp_server.encoding = "utf-8"
        self.connect_status = ""
        self.login_status = ""
        self.connect(HOSTNAME, PORT, USERNAME, PASSWORD)

    def connect(self, hostname: str, port: int, username: str, password: str):
        self.connect_status = self.ftp_server.connect(hostname, port)
        print(f"FTP connect  to {hostname}: {self.connect_status}")
        self.login_status = self.ftp_server.login(username, password)
        print(f"FTP login by {username}: {self.login_status}")

    def upload_file(self, file_path: str, file: BinaryIO) -> bool:
        try:
            self.ftp_server.storbinary(f"STOR {file_path}", file)
            return True
        except Exception as e:
            logging.error(f"FTP write to {file_path} failed. Error: {e}")
            return False

    def download_file(self, file_path: str, file: BinaryIO) -> bool:
        try:
            self.ftp_server.retrbinary(f"RETR {file_path}", file.write)
            return True
        except Exception as e:
            logging.error(f"FTP write to {file_path} failed. Error: {e}")
            return False

    def close(self):
        self.ftp_server.close()


ftpTransfer = _FtpConnector()
