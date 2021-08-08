"""
Test file for model evaluation
"""

import os
import pytest
import logging
import pandas as pd

from sklearn.linear_model import LogisticRegression

from churn.config import CURRENT_DIR, RESULTS_OUTPUT_DIR
from churn.utils import train_models, perform_feature_engineering
from churn.EvaluateModel import EvaluateModel


@pytest.fixture
def data():
    data_path = os.path.join(CURRENT_DIR, 'tests/data/bank_data_sample.csv')
    df = pd.read_csv(data_path)
    return perform_feature_engineering(df, 'Churn', 0.3, 42)


def test_evaluate_model(data):
    """
    Test for model evaluation which:
    1 - run a classification report from scikit learn and store result
    2 - compute roc curve for model and store result
    3 - compute shapley values and store result
    4 - compute feature importances of model and store result
    """
    try:
        output_dir = 'output_test'
        output_result = ['classification_report.csv',
                         'features_importances.png',
                         'roc_curve.png',
                         'shapley_values.png']
        full_output_dir = os.path.join('tests', output_dir, RESULTS_OUTPUT_DIR)
        (X_train, X_test, y_train, y_test), feature_names = data
        lrc = LogisticRegression(max_iter=200)
        fitted_model, predictions = train_models(X_train, X_test, y_train, lrc)
        job_evaluation = EvaluateModel()
        job_evaluation.evaluate_model(fitted_model, X_train, X_test, y_test, predictions, full_output_dir)
        for end_filename in output_result:
            assert os.path.isfile(f'{CURRENT_DIR}/tests/{output_dir}/'
                                  f'{RESULTS_OUTPUT_DIR}/{type(lrc).__name__}_{end_filename}')
            logging.info(f'TEST SUCCESS - during model evualuation : {end_filename} has been stored')
    except AssertionError as err:
        logging.info(f'TEST ERROR - during model evaluation : {err}')

