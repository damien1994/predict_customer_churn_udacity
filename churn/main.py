import logging as lg

from churn.config import IMAGES_EDA_OUTPUT_DIR, IMAGES_EDA_SUB_DIR, NUM_COLUMNS, CAT_COLUMNS, TARGET_COL
from churn.utils import import_data, perform_eda


def main():
    df = import_data("data/bank_data.csv", NUM_COLUMNS, CAT_COLUMNS, TARGET_COL)
    lg.info(df.shape)
    perform_eda(df, NUM_COLUMNS, CAT_COLUMNS, TARGET_COL[0], IMAGES_EDA_OUTPUT_DIR, IMAGES_EDA_SUB_DIR)


if __name__ == '__main__':
    main()
