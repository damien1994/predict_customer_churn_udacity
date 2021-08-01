"""
File to store all functions for customer churn predictions
"""
import logging as lg
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

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
        lg.info('File not found')
        raise err

'''
def perform_eda(df, col):
   plt.figure(figsize=(20, 10))
   if isinstance(col, object):
       perform_eda_on_cat_columns()
   else:
       perform_eda_on_num_columns()
'''


#def perform_eda_on_cat_columns(df, cat_col):
# def performn_eda_on_num_columns(df, num_col)


def encode_target(df: pd.DataFrame, target_name: str, target_encoding: dict) -> dict:
    """
    Encode target variable
    :param df: a pandas dataframe
    :param target_name: the target column to encode
    :param target_encoding: how to encode the target value
    :return: a dictionary with <new_encode : value to be encoded>
    """
    for key, value in target_encoding.items():
        return {key: df[target_name] == value}


def encoder_helper(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """
    Feature engineering - compute a target encoding
    :param df: the pandas dataframe
    :param feature: a column from the dataframe
    :return: a pandas dataframe with the target encoding to the feature
    """
    try:
        feature_groups = df.groupby(feature).mean()['Churn']
        df['{0}_Churn'.format(feature)] = df[feature].map(feature_groups)
        return df
    except:
        lg.info("Something went wrong")


def perform_feature_engineering(df, target, test_size, random_state):
    """
    Split data into train and test labels
    :param df: a pandas dataframe
    :param target: target column
    :param test_size: size of test sample
    :param random_state: controls the randomness of the sampled classes
    :return: data and labels for train and test samples - return pandas dataframes & series
    """
    return train_test_split(df.drop(target, axis=1), df[target], test_size=test_size, random_state=random_state)


def train_models(x_train_data, x_test_data, y_train_data, model, param_grid=None, cv=None,
                       grid_search=False, do_probabilities=False):
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
            fitted_model= model.fit(x_train_data, y_train_data)

        predictions = fitted_model.predict_proba(x_test_data) if do_probabilities else fitted_model.predict(x_test_data)
        return fitted_model, predictions
    except:
        lg.info('complicated life')