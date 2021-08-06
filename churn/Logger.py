"""
Python class to define logger config
"""
import os
import logging


class BaseLogger:
    """
    Base Logger class to initialize logger mode
    """

    def __init__(self):
        self.logger = self.get_logger(self.__module__)

    @staticmethod
    def get_logger(name):
        logging.basicConfig(
            filename=os.path.join(os.path.dirname(__file__), './logs/churn_library.log'),
            level=logging.INFO,
            filemode='w',
            format='%(name)s - %(levelname)s - %(message)s')
        return logging.getLogger()
