"""
Main file where the customer churn prediction is computed
"""
import os
import logging as lg

from churn.config import NUM_COLUMNS, CAT_COLUMNS, TARGET_COL, ENCODING_TARGET
from churn.CustomerChurn import CustomerChurn


def main():
    path_data = os.path.join(os.path.dirname(__file__), "data/bank_data.csv")
    churn_prediction = CustomerChurn(path_data, NUM_COLUMNS, CAT_COLUMNS, TARGET_COL, ENCODING_TARGET)
    lg.info("object churn prediction")
    churn_prediction.compute_customer_churn_predictions()


if __name__ == '__main__':
    main()
