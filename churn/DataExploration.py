"""
Python class for performing EDA on data
author: Damien Michelle
date: 09/08/2021
"""
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from churn.config import CURRENT_DIR
from churn.utils import safe_creation_directory
from churn.base_logger import logging


class DataExploration:

    def __init__(self, num_cols: list, cat_cols: list, target_col: str):
        # super(DataExploration, self).__init__()
        # super().__init__()
        # super(BaseLogger, self).__init__()
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.target_col = target_col

    def perform_eda(self, df, output_dir, sub_dir):
        """
           Perform exploratory data analysis for num, cat and target col
           :param df: a pandas dataframe
           :param output_dir: path to store eda result
           :param sub_dir: sub directories split for univariate/bivariate and num/cat analysis
           """
        try:
            plt.figure(figsize=(20, 10))
            for subdir in sub_dir:
                path = os.path.join(output_dir, subdir)
                safe_creation_directory(path)
            full_output_dir = os.path.join(CURRENT_DIR, output_dir)
            for col in self.num_cols:
                self.univariate_num_analysis(df, col, full_output_dir)
                self.bivariate_num_analysis(df, col, self.target_col[0], 'poly', full_output_dir)
            for col in self.cat_cols:
                self.univariate_cat_analysis(df, col, full_output_dir)
                self.bivariate_cat_analysis(df, col, self.target_col[0], full_output_dir)
            self.compute_correlation_matrix(df, full_output_dir)
            self.plot_target_distribution(df, self.target_col[0], full_output_dir)
            logging.info(f'SUCCESS - EDA has been done entirely !')
        except Exception as err:
            logging.info(f'ERROR - A step has failed during EDA : {err}')

    @staticmethod
    def univariate_cat_analysis(df: pd.DataFrame, feature: str, output_dir: str):
        """
        Plot distribution of discrete columns and save result
        :param df: A pandas dataframe
        :param feature: categorical column to analyse
        :param output_dir: path to store result
        """
        try:
            cat_plot = sns.histplot(df[feature])
            cat_plot.figure.savefig(f'{output_dir}/univarite_cat_analysis/{feature}_histplot_distribution.png',
                                    bbox_inches='tight')
            logging.info(f'SUCCESS - categorical univariate analysis on {feature} has been saved')
        except (AttributeError, NameError, IsADirectoryError, FileNotFoundError, IndexError) as err:
            logging.info(f'ERROR - during univariate analysis on categorical {feature}: {err}')

    @staticmethod
    def univariate_num_analysis(df: pd.DataFrame, feature: str, output_dir: str):
        """
        Plot distribution of continuous columns and save result
        :param df: A pandas dataframe
        :param feature: numerical column to analyse
        :param output_dir: path to store result
        """
        try:
            num_plot = sns.displot(df[feature])
            num_plot.savefig(f'{output_dir}/univariate_num_analysis/{feature}_distplot_distribution.png',
                             bbox_inches='tight')
            logging.info(f'SUCCESS - numerical univariate analysis on {feature} has been saved')
        except (AttributeError, NameError, IsADirectoryError, FileNotFoundError, IndexError) as err:
            logging.info(f'ERROR - during univariate analysis on numerical {feature}: {err}')

    @staticmethod
    def bivariate_cat_analysis(df, feature, target, output_dir):
        """
        Bivariate analysis of numerical columns with target col
        :param df: a pandas dataframe
        :param feature: numerical column to cross with target col
        :param target: target column
        :param output_dir: path to store result
        """
        try:
            cat_bivariate_plot = sns.catplot(x=feature, hue=target, data=df, kind='count')
            cat_bivariate_plot.savefig(f'{output_dir}/bivariate_cat_analysis/{feature}_bivariate_cat_analysis.png',
                                       bbox_inches='tight')
            logging.info(f'SUCCESS - categorical bivariate analysis on {feature} and {target} has been saved')
        except (AttributeError, NameError, IsADirectoryError, FileNotFoundError, IndexError) as err:
            logging.info(f'ERROR - during bivariate analysis on categorical {feature} and {target}: {err}')

    @staticmethod
    def bivariate_num_analysis(df, feature, target, element, output_dir):
        """
        Bivariate analysis of numerical columns with target col
        :param df: a pandas dataframe
        :param feature: numerical column to cross with target col
        :param target: target column
        :param element: figure style with seaborn
        :param output_dir: path to store result
        """
        try:
            num_bivariate_plot = sns.histplot(df, x=feature, hue=target, element=element)
            num_bivariate_plot.figure.savefig(f'{output_dir}/bivariate_num_analysis/'
                                              f'{feature}_bivariate_num_analysis.png', bbox_inches='tight')
            logging.info(f'SUCCESS - numerical bivariate analysis on {feature} and {target} has been saved')
        except (AttributeError, NameError, IsADirectoryError, FileNotFoundError, IndexError) as err:
            logging.info(f'ERROR - during bivariate analysis on numerical {feature} and {target}: {err}')

    @staticmethod
    def compute_correlation_matrix(df: pd.DataFrame, output_dir: str):
        """
        Compute correlation matrix
        :param df: a pandas dataframe
        :param output_dir: path to store result
        """
        try:
            correlation_matrix = sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
            correlation_matrix.figure.savefig(f'{output_dir}/bivariate_num_analysis/correlation_matrix.png',
                                              bbox_inches='tight')
            logging.info(f'SUCCESS - correlation matrix analysis has been saved')
        except (AttributeError, NameError, IsADirectoryError, FileNotFoundError, IndexError) as err:
            logging.info(f'ERROR - during correlation matrix analysis : {err}')

    @staticmethod
    def plot_target_distribution(df: pd.DataFrame, target: str, output_dir: str):
        """
        Compute target distribution
        :param df: a pandas dataframe
        :param target: target to analyze
        :param output_dir: path to store result
        """
        try:
            plt.title('target distribution')
            target_distribution = df[target].value_counts('normalize').plot(kind='bar')
            plt.savefig(f'{output_dir}/target/target_distribution.png', bbox_inches='tight')
            logging.info(f'SUCCESS - target distribution plot has been saved')
        except (AttributeError, NameError, IsADirectoryError, FileNotFoundError, IndexError) as err:
            logging.info(f'ERROR - during target distribution analysis : {err}')
