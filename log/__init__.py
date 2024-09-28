import logging

logger = logging.getLogger("sqlalchemy")
logger.setLevel(logging.ERROR)
# logger = logging.getLogger("enhancement")

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)
file_handler = logging.FileHandler("enhancement.log")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)

logger.addHandler(console_handler)
logger.addHandler(file_handler)
