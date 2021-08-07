"""
Python class for evaluating model performance
"""
import os
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report

from churn.config import CURRENT_DIR
from churn.utils import safe_creation_directory
from churn.base_logger import logging


class EvaluateModel:
    """
    Evaluate model performance and store results into a subfolder
    Steps performed:
    - classification report
    - compute roc curve
    - compute shapley values
    - compute features importances
    """

    def __init__(self):
        super(EvaluateModel, self).__init__()

    def evaluate_model(self, model_fitted, train_data: pd.DataFrame, test_data: pd.DataFrame,
                       true_labels: pd.Series, predictions: np.array, output_dir: str):
        """
        Function that calls all actions for model evaluation
        :param model_fitted: a scikit learn object (for instance) - the model fitted
        :param train_data: data on which model has been trained
        :param test_data: test data that will allow us to make the predictions
        :param true_labels: true values to compare with predictions
        :param predictions: predictions made with the model trained
        :param output_dir: directory where to store result
        :return : a lot of plots to contextualize model performance
        """
        try:
            if isinstance(model_fitted, GridSearchCV):
                model_name = type(model_fitted.best_estimator_).__name__
                coefficients = model_fitted.best_estimator_.feature_importances_
            else:
                model_name = type(model_fitted).__name__
                coefficients = model_fitted.coef_[0]
            safe_creation_directory(output_dir)
            full_output_dir = os.path.join(CURRENT_DIR, output_dir)
            self.store_classification_report(true_labels, predictions, model_name, full_output_dir)
            logging.info("SUCCESS - Compute classification report")
            self.store_model_roc_curve(model_fitted, test_data, true_labels, model_name, full_output_dir)
            logging.info("SUCCESS - Compute roc curve")
            self.compute_shapley_values(model_fitted, test_data, full_output_dir, model_name, train_data)
            logging.info("SUCCESS - Compute Shapley values")
            self.compute_features_importances(train_data.columns, coefficients, model_name, full_output_dir)
            logging.info("SUCCESS - Compute feature importance")
        except (AssertionError, AttributeError, IsADirectoryError, NotADirectoryError, SyntaxError, ValueError) as err:
            logging.info(f'ERROR - during model evaluation : {err}')

    @staticmethod
    def store_classification_report(true_labels, predictions, model_name, output_dir):
        """
        Compute classification report thanks to scikit learn function.
        Returns some metrics like precision, recall, accuracy, ...
        and save it into a csv file
        :param true_labels: true values to compare with predictions
        :param predictions: predictions made with the model trained
        :param model_name: name of the model used like 'LogisticRegression' or 'RandomForest'
        :param output_dir: directory where to store result
        :returns: a csv file where report result is stored
        """
        try:
            report = classification_report(true_labels, predictions, output_dict=True)
            return pd.DataFrame(report).transpose().to_csv(f'{output_dir}/{model_name}_classification_report.csv')
        except (ValueError, TypeError, FileNotFoundError) as err:
            logging.info(f'ERROR - during classification report compute : {err}')

    @staticmethod
    def store_model_roc_curve(model_fitted, test_data, true_labels, model_name, output_dir):
        """
        Compute roc curve and save it into a png file
        :param model_fitted: the model fitted
        :param test_data: test data that will allow us to make the predictions
        :param true_labels:  true values to compare with predictions
        :param model_name: name of the model used like 'LogisticRegression' or 'RandomForest'
        :param output_dir: directory where to store result
        :returns: a png file where roc curve is stored
        """
        try:
            plt.figure(figsize=(15, 8))
            ax = plt.gca()
            plot_roc_curve(model_fitted, test_data, true_labels, ax=ax, alpha=0.8)
            plt.savefig(f'{output_dir}/{model_name}_roc_curve.png', bbox_inches='tight')
        except (ValueError, TypeError, FileNotFoundError) as err:
            logging.info(f'ERROR - during model roc curve compute : {err}')

    @staticmethod
    def compute_shapley_values(model_fitted, test_data, output_dir, model_name, train_data=None):
        """
        Compute shapley values and save them into a png file
        :param model_fitted: the model fitted
        :param test_data: test data that will allow us to make the predictions
        :param output_dir: directory where to store result
        :param model_name: name of the model used like 'LogisticRegression' or 'RandomForest'
        :param train_data: data on which model has been trained
        :returns a png file with shapley values computed
        """
        try:
            explainer = None
            plt.figure(figsize=(20, 10))
            if isinstance(model_fitted, GridSearchCV):
                model_fitted = model_fitted.best_estimator_
            if model_name in ['LogisticRegression', 'LinearRegression']:
                masker = shap.maskers.Independent(data=train_data)
                explainer = shap.LinearExplainer(model_fitted, masker=masker)
            elif model_name in ['RandomForestClassifier']:
                explainer = shap.TreeExplainer(model_fitted)
            else:
                logging.info('Add your model type to tree, linear, gradient or deep explainer '
                             'and add condition to this function')
            shap_values = explainer.shap_values(test_data)
            shap.summary_plot(shap_values, test_data, show=False)
            plt.savefig(f'{output_dir}/{type(model_fitted).__name__}_shapley_values.png', bbox_inches='tight')
        except (AssertionError, AttributeError, ValueError, TypeError, FileNotFoundError) as err:
            logging.info(f'ERROR - during shapley values compute: {err}')

    @staticmethod
    def compute_features_importances(feature_names: pd.Index, coefficients: np.array, model_name: str, output_dir: str):
        """
        Compute features importances and save them into a png file
        :param feature_names: column names used for training
        :param coefficients: coefficients of each column used in training which corresponds to their weight
        in the prediction
        :param model_name: name of the model used like 'LogisticRegression' or 'RandomForest'
        :param output_dir: directory where to store result
        :returns a png file with features importances computed
        """
        try:
            plt.figure(figsize=(20, 10))
            safe_creation_directory(output_dir)
            df_features_importances = pd.DataFrame(zip(feature_names, coefficients),
                                                   columns=['feature', 'coefficient']).sort_values(by=['coefficient'],
                                                                                                   ascending=False)
            # Plot Searborn bar chart
            sns.barplot(x=df_features_importances['coefficient'], y=df_features_importances['feature'])
            # Add chart labels
            plt.title(model_name + 'FEATURE IMPORTANCE')
            plt.xlabel('FEATURE IMPORTANCE')
            plt.ylabel('FEATURE NAMES')
            plt.savefig(f'{output_dir}/{model_name}_features_importances.png', bbox_inches='tight')
        except (ValueError, TypeError, FileNotFoundError) as err:
            logging.info(f'ERROR - during features importance compute: {err}')
