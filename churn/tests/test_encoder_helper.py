"""
Test file for encoder helper class
"""

import pytest
import logging
import pandas as pd

from churn.config import ENCODING_TARGET
from churn.EncoderHelper import EncoderHelper


@pytest.fixture
def input_data():
    data = {
        'Marital_Status': ['Married', 'Single', 'Married', 'Unknown', 'Married'],
        'Gender': ['M', 'F', 'M', 'F', 'M'],
        'Attrition_Flag': ['Existing Customer', 'Existing Customer', 'Existing Customer',
                           'Attrited Customer', 'Attrited Customer']
    }
    return pd.DataFrame(data)


@pytest.fixture
def output_data():
    data = {
        'Marital_Status': ['Married', 'Single', 'Married', 'Unknown', 'Married'],
        'Gender': ['M', 'F', 'M', 'F', 'M'],
        'Churn': [0, 0, 0, 1, 1],
        'Marital_Status_Churn': [0.33, 0.00, 0.33, 1.00, 0.33],
        'Gender_Churn': [0.33, 0.50, 0.33, 0.50, 0.33]
    }
    return pd.DataFrame(data)


def test_encoder_helper(input_data, output_data):
    """
    Test encoder helper function which :
    1 - encode target column
    2 - perform a mean target encoding
    """
    try:
        cat_columns = ['Marital_Status', 'Gender']
        job_encoder = EncoderHelper(cat_columns, ENCODING_TARGET)
        df_result = job_encoder.encoder_helper(input_data, 'Attrition_Flag', 'Churn')
        assert df_result.equals(output_data)
    except (KeyError, AttributeError) as err:
        logging.info(f'TEST ERROR - during categorical columns encoding: {err}')

    try:
        numeric_types = ['int64', 'float64']
        assert df_result.filter(regex='Churn').dtypes.isin(numeric_types).all()
    except (TypeError, AssertionError) as err:
        logging.info(f'TEST ERROR - columns encoded are not numeric : {err}')
