import logging
from datetime import datetime


class Logger:
    def __init__(self, filename="log_{time}.log"):
        format_str = "%(asctime)-24s %(levelname)-8s | %(message)s"
        filename.format(time=datetime.now().strftime("%Y_%m_%d-%H_%M"))

        logging.basicConfig(
            filename=filename,
            level=logging.INFO,
            format=format_str,
            datefmt="%H:%M:%S",
        )

    @staticmethod
    def log(message):
        logging.info(message)

    @staticmethod
    def log_config(config):
        for section, values in config.items():
            for key, value in values.items():
                logging.info(f"{section: <18} -- {key: <24}: {value}")
