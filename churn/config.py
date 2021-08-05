IMAGES_EDA_OUTPUT_DIR = 'images/eda'
IMAGES_EDA_SUB_DIR = [
    'univariate_num_analysis',
    'univarite_cat_analysis',
    'bivariate_cat_analysis',
    'bivariate_num_analysis',
    'target'
]
MODEL_OUTPUT_DIR = 'models'
RESULTS_OUTPUT_DIR = 'results'

CAT_COLUMNS = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]

NUM_COLUMNS = [
    'Customer_Age',
    'Dependent_count',
    'Months_on_book',
    'Total_Relationship_Count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio'
]

TARGET_COL = [
    'Attrition_Flag'
]

ENCODING_TARGET = {
    0: 'Existing Customer',
    1: 'Attrited Customer'
}

PARAM_GRID = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [4, 5, 100],
    'criterion': ['gini', 'entropy']
}
