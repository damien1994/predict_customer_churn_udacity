"""
Test file for feature engineering
"""
import pytest
import pandas as pd
import numpy as np

from churn.base_logger import logging
from churn.utils import perform_feature_engineering


@pytest.fixture
def input_data():
    data = {
        'Customer_Age': [45, 49, 51, 40, 40, 44, 51, 32, 37, 48],
        'Gender_Churn': [0.33, 0.50, 0.33, 0.50, 0.33, 0.50, 0.33, 0.33, 0.50, 0.50],
        'Gender': ['M', 'F', 'M', 'F', 'M', 'M', 'F', 'M', 'F', 'F'],
        'Churn': [0, 1, 0, 1, 0, 0, 0, 1, 1, 0]
    }
    return pd.DataFrame(data)


def test_perform_feature_engineering(input_data):
    """
    Test perform feature engineering function which :
    1 - delete all non numeric columns for modelisation
    2 - split data into train/test and X/y samples
    3 - keep columns names for posterior model evaluation
    """
    try:
        (X_train, X_test, y_train, y_test), feature_names = perform_feature_engineering(input_data, 'Churn', 0.3, 42)
        assert len(feature_names) == 2
        assert X_train.shape == (7, 2)
        assert X_test.shape == (3, 2)
        assert (type(y_train), type(y_test)) == (pd.Series, pd.Series)
        assert (len(y_train), len(y_test)) == (7, 3)
        logging.info(f'TEST SUCCESS - performing feature engineering')
    except (AssertionError, KeyError) as err:
        logging.info(f'TEST ERROR - during performing feature engineering : {err}')
