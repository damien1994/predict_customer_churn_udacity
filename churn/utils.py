"""
File to store all functions for customer churn predictions
"""
import os
import logging as lg
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report


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


def perform_eda(df: pd.DataFrame, num_cols: list, cat_cols: list, target_col: str, output_dir: str, sub_dir: str):
    """
    Perform exploratory data analysis for num, cat and target col
    :param df: a pandas dataframe
    :param num_cols: list of numerical columns
    :param cat_cols: list of categorical columns
    :param target_col: the target column
    :param output_dir: path to store eda result
    :param sub_dir: sub directories split for univariate/bivariate and num/cat analysis
    """
    plt.figure(figsize=(20, 10))
    for subdir in sub_dir:
        path = os.path.join(output_dir, subdir)
        safe_creation_directory(path)
    for col in num_cols:
        univariate_num_analysis(df, col, output_dir)
        bivariate_num_analysis(df, col, target_col, 'poly', output_dir)
    for col in cat_cols:
        univariate_cat_analysis(df, col, output_dir)
        bivariate_cat_analysis(df, col, target_col, output_dir)
    compute_correlation_matrix(df, output_dir)
    plot_target_distribution(df, target_col, output_dir)


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


def classification_report_image(true_labels, predictions, name_model, mode, output_path):
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    :param true_labels:
    :param predictions:
    :param name_model:
    :output_path:
    :return:
    """
    try:
        lg.info(f'{name_model} results')
        if mode == 'train':
            lg.info(f'{mode} results')
            return classification_report(true_labels, predictions)
    except:
        lg.info('Something went from')


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
            fitted_model = model.fit(x_train_data, y_train_data)

        predictions = fitted_model.predict_proba(x_test_data) if do_probabilities else fitted_model.predict(x_test_data)
        return fitted_model, predictions
    except:
        lg.info('complicated life')


def plot_roc_curves(model, true_labels, predictions):
    """
    Plot roc curve associated to the model
    :param model:
    :param true_labels:
    :param predictions:
    :return:
    """
    try:
        return plot_roc_curve(model, true_labels, predictions)
    except:
        lg.info('try something')


def safe_creation_directory(path):
    """
    Check if directory exists, if not, create it
    :param path: path to store eda results
    """
    if not os.path.isdir(path):
        os.makedirs(path)
        lg.info(f'folder has been created at {path}')
    else:
        lg.info(f'EDA images are stored in {path}')


def univariate_cat_analysis(df: pd.DataFrame, feature: str, output_dir: str):
    """
    Plot distribution of discrete columns and save result
    :param df: A pandas dataframe
    :param feature: categorical column to analyse
    :param output_dir: path to store result
    """
    cat_plot = sns.histplot(df[feature])
    cat_plot.figure.savefig(f'{output_dir}/univarite_cat_analysis/{feature}_histplot_distribution.png', bbox_inches='tight')


def univariate_num_analysis(df: pd.DataFrame, feature: str, output_dir: str):
    """
    Plot distribution of continuous columns and save result
    :param df: A pandas dataframe
    :param feature: numerical column to analyse
    :param output_dir: path to store result
    """
    num_plot = sns.displot(df[feature])
    num_plot.savefig(f'{output_dir}/univariate_num_analysis/{feature}_distplot_distribution.png', bbox_inches='tight')


def bivariate_cat_analysis(df, feature, target, output_dir):
    """
    Bivariate analysis of numerical columns with target col
    :param df: a pandas dataframe
    :param feature: numerical column to cross with target col
    :param target: target column
    :param output_dir: path to store result
    """
    cat_bivariate_plot = sns.catplot(x=feature, hue=target, data=df, kind='count')
    cat_bivariate_plot.savefig(f'{output_dir}/bivariate_cat_analysis/{feature}_bivariate_cat_analysis.png', bbox_inches='tight')


def bivariate_num_analysis(df, feature, target, element, output_dir):
    """
    Bivariate analysis of numerical columns with target col
    :param df: a pandas dataframe
    :param feature: numerical column to cross with target col
    :param target: target column
    :param element: figure style with seaborn
    :param output_dir: path to store result
    """
    num_bivariate_plot = sns.histplot(df, x=feature, hue=target, element=element)
    num_bivariate_plot.figure.savefig(f'{output_dir}/bivariate_num_analysis/{feature}_bivariate_num_analysis.png', bbox_inches='tight')


def compute_correlation_matrix(df: pd.DataFrame, output_dir: str):
    """
    Compute correlation matrix
    :param df: a pandas dataframe
    :output_dir: path to store result
    """
    correlation_matrix = sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    correlation_matrix.figure.savefig(f'{output_dir}/bivariate_num_analysis/correlation_matrix.png', bbox_inches='tight')


def plot_target_distribution(df: pd.DataFrame, target: str, output_dir: str):
    """
    Compute target distribution
    :param df: a pandas dataframe
    :param target: target to analyze
    :param output_dir: path to store result
    """
    plt.title('target distribution')
    target_distribution = df[target].value_counts('normalize').plot(kind='bar')
    plt.savefig(f'{output_dir}/target/target_distribution.png', bbox_inches='tight')
