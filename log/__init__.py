import logging

logger = logging.getLogger("enhancement")
logger.setLevel(logging.INFO)
# create file handler which logs even debug messages
fh = logging.FileHandler("enhancement.log")
fh.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)

logger.addHandler(fh)
