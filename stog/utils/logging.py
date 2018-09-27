from __future__ import absolute_import

import logging



logger = logging.getLogger()


def init_logger(log_file=None):
    """
    Adopted from OpenNMT-py:
        https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/utils/logging.py
    """
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger