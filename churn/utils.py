"""
File to store all functions for customer churn predictions
"""
import os
import argparse
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from churn.config import CURRENT_DIR
from churn.base_logger import logging


def create_parser():
    """
    Parser
    :return: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_file',
        help='csv file to process',
        required=True
    )
    return parser


def parse_args(args):
    """
    Parse arguments
    :param args: raw args
    :return: Parsed arguments
    """
    parser = create_parser()
    return parser.parse_args(args=args)


def import_data(input_path: str, num_columns: list, cat_columns: list, target_column: list) -> pd.DataFrame:
    """
    Read csv file
    :param input_path: path to csv file
    :param num_columns: columns names for numeric variables
    :param cat_columns: columns names for categorical variables
    :param target_column: target column
    :return: a pandas dataframe without unnamed columns
    """
    try:
        dataframe = pd.read_csv(input_path)
        return dataframe[num_columns + cat_columns + target_column]
    except FileNotFoundError as err:
        logging.info(f'ERROR - during import data: {err}')


def perform_feature_engineering(df: pd.DataFrame, target: str, test_size: float, random_state: int):
    """
    Split data into train and test labels
    :param df: a pandas dataframe
    :param target: target column
    :param test_size: size of test sample
    :param random_state: controls the randomness of the sampled classes
    :return: data and labels for train and test samples - return pandas dataframes & series
    """
    try:
        df = df.select_dtypes(exclude=['object'])
        return train_test_split(df.drop(target, axis=1), df[target], test_size=test_size, random_state=random_state), \
               df.drop(target, axis=1).columns
    except KeyError as error:
        logging.info(f'ERROR - during feature engineering: {error}')


def train_models(x_train_data: pd.DataFrame, x_test_data: pd.DataFrame, y_train_data: pd.Series, model,
                 param_grid=None, cv=None, grid_search=False, do_probabilities=False, output_dir=None):
    """
    Model training function. You can put a grid search on your model
    and return probabilities if you want
    :param x_train_data: data used for model training - a pandas dataframe
    :param x_test_data: data used for predictions - a pandas dataframe
    :param y_train_data: true labels - a pandas series
    :param model: a scikit learn classifier or regressor (could be something else)
    :param param_grid: grid with search space for tuning hyperparameters during modelisation - list of values
    :param cv: number of folds for cross validation
    :param grid_search: boolean - defines if you want to use a grid search for parameters hypertuning
    :param do_probabilities: boolean - defines if you prefer probabilities or binary class as predictions
    :param output_dir: directory where to store the model in pkl format if desire
    :returns: the model fitted and the predictions
    """
    try:
        if grid_search:
            gs = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=cv,
                n_jobs=-1,
                verbose=2
            )
            fitted_model = gs.fit(x_train_data, y_train_data)
        else:
            fitted_model = model.fit(x_train_data, y_train_data)

        logging.info(f'SUCCESS - model has been fitted')

        if output_dir:
            save_model(fitted_model, output_dir)
            logging.info(f'SUCCESS - Model has been saved to {output_dir}')

        predictions = fitted_model.predict_proba(x_test_data) if do_probabilities else fitted_model.predict(x_test_data)
        return fitted_model, predictions
    except TypeError as err:
        logging.info(f'ERROR during model training: {err}')


def safe_creation_directory(path):
    """
    Check if directory exists, if not, create it
    :param path: path to store eda results
    """
    try:
        full_path = os.path.join(CURRENT_DIR, path)
        if not os.path.isdir(full_path):
            os.makedirs(full_path)
            logging.info(f'SUCCESS - folder has been created at {full_path}')
        else:
            logging.info(f'SUCCESS - EDA images are stored in {full_path}')
    except (OSError, SyntaxError) as err:
        logging.info(f'ERROR - during directory creation: {err}')


def save_model(model, output_dir: str):
    """
    Save the model fitted into a .pkl file
    :param model: model fitted - an scikit learn object in our case
    :param output_dir: path where to store the model
    :return a .pkl into the output path
    """
    try:
        safe_creation_directory(output_dir)
        return joblib.dump(model, f'{output_dir}/{type(model).__name__}.pkl')
    except Exception as err:
        logging.info(f'ERROR - during model dump: {err}')


def load_model(input_path: str):
    """
    Load a .pkl file into a model
    :param input_path: path where the model is stored
    :return a a scikit learn model (in our case - can return other objects)
    """
    try:
        return joblib.load(input_path)
    except (FileNotFoundError, MemoryError) as err:
        logging.info(f'ERROR - during model loading: {err}')
