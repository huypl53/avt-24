import logging
import os

log_level = os.getenv("LOG_LEVEL", "ERROR").upper()

logger = logging.getLogger()  # logging.getLogger("sqlalchemy")
# logger.setLevel(logging.ERROR)
# logger = logging.getLogger("enhancement")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler = logging.StreamHandler()
console_handler.setLevel(log_level)

file_handler = logging.FileHandler("root.log")
file_handler.setFormatter(formatter)
file_handler.setLevel(log_level)

logger.addHandler(console_handler)
logger.addHandler(file_handler)
