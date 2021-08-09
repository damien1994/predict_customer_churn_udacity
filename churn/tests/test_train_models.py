"""
Test file for train model
author: Damien Michelle
date: 09/08/2021
"""
import os
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
import pandas as pd

from sklearn.linear_model import LogisticRegression

from churn.config import CURRENT_DIR
from churn.utils import train_models, perform_feature_engineering
from churn.base_logger import logging


@pytest.fixture
def data():
    data_path = os.path.join(CURRENT_DIR, 'tests/data/bank_data_sample.csv')
    df = pd.read_csv(data_path)
    return perform_feature_engineering(df, 'Churn', 0.3, 42)


@pytest.mark.filterwarnings("ignore: lbfgs failed to converge")
def test_train_models(data):
    """
    Test for train models function which:
    1 - fit a simple model or a grid search
    2 - [optional] save the model as a .pkl file
    3 - compute predictions from model
    """
    try:
        (X_train, X_test, y_train, y_test), feature_names = data
        lrc = LogisticRegression(max_iter=100)
        fitted_model, predictions = train_models(X_train, X_test, y_train, lrc)
        assert_array_almost_equal(np.round(fitted_model.intercept_, 3), [0.035], decimal=1)
        assert_array_almost_equal(fitted_model.coef_, [[-7.52866129e-03,  3.05745489e-01,  1.91205707e-02,
                                                        -3.69219615e-01,  5.59305942e-01,  6.92126822e-01,
                                                        -3.00673620e-04, -5.96828369e-04,  2.96154754e-04,
                                                        -1.71901823e-01,  3.03169297e-04, -8.64173480e-02,
                                                        -2.21152249e-01, -7.52902453e-03,  9.73271897e-03,
                                                        8.18308479e-03,  9.75816419e-03,  7.38483958e-03,
                                                        6.83536439e-03]], decimal=1)
        assert len(predictions) == 300
        assert max(predictions), min(predictions) == (0, 1)
        logging.info(f'TEST SUCCESS - model correctly fitted')
    except AssertionError as err:
        logging.info(f'TEST ERROR - during model training : {err}')

