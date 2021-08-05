import logging as lg

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from churn.config import IMAGES_EDA_OUTPUT_DIR, IMAGES_EDA_SUB_DIR, MODEL_OUTPUT_DIR, \
    RESULTS_OUTPUT_DIR, NUM_COLUMNS, CAT_COLUMNS, TARGET_COL, PARAM_GRID, ENCODING_TARGET
from churn.utils import import_data, perform_eda, encoder_helper, perform_feature_engineering, \
    train_models, evaluate_model


def main():
    lg.info('Import data')
    df = import_data("data/bank_data.csv", NUM_COLUMNS, CAT_COLUMNS, TARGET_COL)
    lg.info(df.shape)
    lg.info('Perform EDA')
    #perform_eda(df, NUM_COLUMNS, CAT_COLUMNS, TARGET_COL[0], IMAGES_EDA_OUTPUT_DIR, IMAGES_EDA_SUB_DIR)
    lg.info('Perform basic encoding')
    df = encoder_helper(df, 'Attrition_Flag', 'Churn', ENCODING_TARGET, CAT_COLUMNS)
    print(df.shape)
    lg.info('Split data into train/test and x/y datasets')
    (X_train, X_test, y_train, y_test), feature_names = perform_feature_engineering(df, 'Churn', 0.3, 42)
    print(X_train.shape, X_test.shape, len(y_train), len(y_test))
    lg.info('Perform modelisation')
    lrc = LogisticRegression(max_iter=200)
    rfc = RandomForestClassifier(random_state=42)
    for model in [lrc, rfc]:
        if model == lrc:
            fitted_model, predictions = train_models(X_train, X_test, y_train, model, output_dir=MODEL_OUTPUT_DIR)
        elif model == rfc:
            fitted_model, predictions = train_models(X_train, X_test, y_train, model, param_grid=PARAM_GRID,
                                                     cv=5, grid_search=True, do_probabilities=False,
                                                     output_dir=MODEL_OUTPUT_DIR)
        lg.info('Perform model evaluation')
        # TO DO : insert an argument to choose if you want to train model or load their
        # pickle format
        #fitted_model = lrc.fit(X_train, y_train)
        #predictions = fitted_model.predict(X_test)
        evaluate_model(fitted_model, X_train, X_test, y_test, predictions, RESULTS_OUTPUT_DIR)


if __name__ == '__main__':
    main()
