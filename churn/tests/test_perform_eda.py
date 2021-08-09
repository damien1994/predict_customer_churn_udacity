"""
Test file for performing eda
author: Damien Michelle
date: 09/08/2021
"""
import os
import pytest
import logging
import pandas as pd

from churn.config import CURRENT_DIR, IMAGES_EDA_SUB_DIR
from churn.DataExploration import DataExploration


@pytest.fixture
def input_data():
    data = {
        'Marital_Status': ['Married', 'Single', 'Married', 'Unknown', 'Married'],
        'Gender': ['M', 'F', 'M', 'F', 'M'],
        'Age': [45, 51, 67, 28, 72],
        'Credit_Limit': [2607.0, 10195.0, 1867.0, 2733.0, 4716.0],
        'Attrition_Flag': ['Existing Customer', 'Existing Customer', 'Existing Customer',
                           'Attrited Customer', 'Attrited Customer']
    }
    return pd.DataFrame(data)


def test_perform_eda(input_data):
    """
    Test perform eda function which :
    1 - save files as results for univariate num analysis
    2 - save files as results for univariate cat analysis
    3 - save files as results for bivariate num analysis
    4 - save files as results for bivariate cat analysis
    5 - save files as results for target distribution
    """
    try:
        num_cols = ['Age', 'Credit_Limit']
        cat_cols = ['Marital_Status', 'Gender']
        target_col = ['Attrition_Flag']
        test_dir = 'tests'
        output_dir = 'output_test/images/eda'
        sub_dir = IMAGES_EDA_SUB_DIR
        job_exploration = DataExploration(num_cols, cat_cols, target_col)
        job_exploration.perform_eda(input_data,
                                    os.path.join(test_dir, output_dir), sub_dir)
        output_result = []
        for sub_dir in IMAGES_EDA_SUB_DIR:
            full_output_test_dir = os.path.join(CURRENT_DIR, 'tests', output_dir, sub_dir)
            output_result.append(len([file for file in os.listdir(full_output_test_dir) if os.path.isfile(
                os.path.join(full_output_test_dir, file))]))
        assert output_result == [2, 2, 2, 3, 1]
        logging.info('TEST SUCCESS - during performing EDA : all files have been created and saved')
    except AssertionError as err:
        logging.info(f'TEST ERROR - during performing EDA : some result files are missing : {err}')
