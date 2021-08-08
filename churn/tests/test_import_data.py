"""
Test file for data import
"""
import os
import logging
import pytest
import pandas as pd

from churn.config import CURRENT_DIR, NUM_COLUMNS, CAT_COLUMNS, TARGET_COL
from churn.utils import import_data
#from churn.tests.base_logger import logging


def test_import():
    """
    test data import - this example is completed for you to assist with the other test functions
    """
    try:
        data_path = os.path.join(CURRENT_DIR, 'data/bank_data.csv')
        df = import_data(data_path, NUM_COLUMNS, CAT_COLUMNS, TARGET_COL)
        logging.info('TEST SUCCESS - import_data function')
    except FileNotFoundError as err:
        logging.error(f'TEST ERROR - during import data - The file was not found : {err}')

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        assert df.shape[1] == len(NUM_COLUMNS + CAT_COLUMNS + TARGET_COL)
        logging.info('TEST SUCCESS - dataframe seems to have right dimensions')
    except (AssertionError, IndexError) as err:
        logging.error(f'TEST ERROR - during import_data: The file does not appear to have rows and columns'
                      f'/ mismatch between number of cols expected and real number of cols : {err}')
