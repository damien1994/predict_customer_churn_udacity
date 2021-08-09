"""
Main file where the customer churn prediction is computed
author: Damien Michelle
date: 09/08/2021
"""
import os
import sys

from churn.utils import parse_args
from churn.base_logger import logging
from churn.config import NUM_COLUMNS, CAT_COLUMNS, TARGET_COL, ENCODING_TARGET
from churn.CustomerChurn import CustomerChurn


def main(input_file):
    """
    Main function where CustomerChurn class is executed to predict churn
    and save all the results
    """
    try:
        churn_prediction = CustomerChurn(input_file, NUM_COLUMNS, CAT_COLUMNS, TARGET_COL, ENCODING_TARGET)
        churn_prediction.compute_customer_churn_predictions()
        logging.info('SUCCESS - All script has been executed with success ! Congratulations !')
    except Exception as err:
        logging.info(f'ERROR - during the execution of the script: {err}')


if __name__ == '__main__':
    ARGS = parse_args(args=sys.argv[1:])
    INPUT_FILE = ARGS.input_file
    main(INPUT_FILE)
