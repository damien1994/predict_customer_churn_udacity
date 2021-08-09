"""
Python class for customer churn prediction
Centralize all steps for churn prediction
author: Damien Michelle
date: 09/08/2021
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from churn.config import IMAGES_EDA_OUTPUT_DIR, IMAGES_EDA_SUB_DIR, MODEL_OUTPUT_DIR, RESULTS_OUTPUT_DIR, PARAM_GRID
from churn.utils import import_data, perform_feature_engineering, train_models
from churn.base_logger import logging
from churn.DataExploration import DataExploration
from churn.EncoderHelper import EncoderHelper
from churn.EvaluateModel import EvaluateModel


class CustomerChurn(DataExploration, EncoderHelper, EvaluateModel):
    """
    Class for customer churn prediction
    """

    def __init__(self, path_data, num_cols, cat_cols, target_col, target_encoding_dict):
        DataExploration.__init__(self, num_cols, cat_cols, target_col)
        EncoderHelper.__init__(self, cat_cols, target_encoding_dict)
        EvaluateModel.__init__(self)
        self.path_data = path_data

    def compute_customer_churn_predictions(self):
        """
        Full data science process for customer churn prediction
        Steps are:
        - data importation
        - EDA
        - encoding
        - feature engineering : split train/test
        - model training
        - model evaluation
        """
        try:
            df = import_data(self.path_data, self.num_cols, self.cat_cols, self.target_col)
            logging.info('SUCCESS - Global data importation has been performed')

            self.perform_eda(df, IMAGES_EDA_OUTPUT_DIR, IMAGES_EDA_SUB_DIR)
            logging.info('SUCCESS - Global EDA step has been performed')

            df = self.encoder_helper(df, 'Attrition_Flag', 'Churn')
            logging.info('SUCCESS - Global encoder helper step has been performed')

            (x_train, x_test, y_train, y_test), feature_names = \
                perform_feature_engineering(df, 'Churn', 0.3, 42)
            logging.info('SUCCESS - Global feature engineering step has been performed')

            lrc = LogisticRegression(max_iter=200)
            rfc = RandomForestClassifier(random_state=42)
            for model in [lrc, rfc]:
                if model == lrc:
                    fitted_model, predictions = \
                        train_models(x_train, x_test, y_train, model,
                                     output_dir=MODEL_OUTPUT_DIR)
                elif model == rfc:
                    fitted_model, predictions = \
                        train_models(x_train, x_test, y_train, model, param_grid=PARAM_GRID,
                                     cv=5, grid_search=True, do_probabilities=False,
                                     output_dir=MODEL_OUTPUT_DIR)
                logging.info('SUCCESS - Global model training step has been performed')

                self.evaluate_model(fitted_model, x_train, x_test, y_test, predictions, RESULTS_OUTPUT_DIR)
                logging.info('SUCCESS - Global model evaluation step has been performed')
        except Exception as err:
            logging.info(f'ERROR - Customer churn prediction has not been performed: {err}')
