import logging

logging.getLogger("sqlalchemy").setLevel(logging.ERROR)

logger = logging.getLogger("enhancement")

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)
# create file handler which logs even debug messages
file_handler = logging.FileHandler("enhancement.log")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)

logger.addHandler(console_handler)
logger.addHandler(file_handler)
