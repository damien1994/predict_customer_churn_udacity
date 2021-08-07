"""
Python class for encoding categorical columns
"""
import pandas as pd
import numpy as np

from churn.base_logger import logging


class EncoderHelper:
    """
    Encoder for categorical columns
    Encode target col and make some encoding engineering
    """

    def __init__(self, cat_cols: list, target_encoding_dict: dict):
        """
        :param cat_cols: list of categorical columns
        :param target_encoding_dict: dict that shows how to encode the target value
        """
        self.cat_cols = cat_cols
        self.target_encoding_dict = target_encoding_dict

    def encoder_helper(self, df, target_name: str, new_target_name: str) -> pd.DataFrame:
        """
        Encode target and categorical features
        :param df: a pandas dataframe
        :param target_name: the target column to encode
        :param new_target_name: name of the target col encoded
        :return: a pandas dataframe with the target and categorical column encoded
        """
        try:
            dataframe = self.encode_target(df, target_name, new_target_name, self.target_encoding_dict)
            logging.info(f'SUCCES - target has been perfectly encoded')
            for col in self.cat_cols:
                self.encoder_cat_features(dataframe, col)
            return dataframe
        except (KeyError, AttributeError) as err:
            logging.info(f'ERROR - during encoding helper : {err}')

    def encode_target(self, df: pd.DataFrame, target_name: str, new_target_name: str,
                      target_encoding_dict: dict) -> pd.DataFrame:
        """
        Encode target variable
        :param df: a pandas dataframe
        :param target_name: the target column to encode
        :param new_target_name: name of the target col encoded
        :param target_encoding_dict: how to encode the target value
        :return: a pandas series that corresponds to the new target column encoded
        """
        try:
            encoding_dict = self.compute_encoding_target_dict(df, target_name, target_encoding_dict)
            df[new_target_name] = np.select(encoding_dict.values(), encoding_dict.keys(), default=1)
            return df.drop(target_name, axis=1)
        except (KeyError, AttributeError, ) as err:
            logging.info(f'ERROR - during target encoding : {err}')

    @staticmethod
    def encoder_cat_features(df: pd.DataFrame, feature: str) -> pd.DataFrame:
        """
        Feature engineering - compute a target encoding
        :param df: the pandas dataframe
        :param feature: a column from the dataframe
        :return: a pandas dataframe with the target encoding to the feature
        """
        try:
            feature_groups = round(df.groupby(feature).mean()['Churn'], 2)
            df['{0}_Churn'.format(feature)] = df[feature].map(feature_groups)
            logging.info(f'SUCCESS - {feature} has been encoded')
            return df
        except (KeyError, AttributeError) as err:
            logging.info(f'ERROR - during categorical columns encoding : {err}')

    @staticmethod
    def compute_encoding_target_dict(df: pd.DataFrame, target_name: str, target_encoding_dict: dict) -> dict:
        """
        Compute dictionary for encoding target
        :param df: a pandas dataframe
        :param target_name: the target column to encode
        :param target_encoding_dict: how to encode the target value
        :return: a dictionary with <new_encode : value to be encoded>
        """
        try:
            for key, value in target_encoding_dict.items():
                return {key: df[target_name] == value}
        except (KeyError, AttributeError) as err:
            logging.info(f'ERROR - during compute of target encoding : {err}')
