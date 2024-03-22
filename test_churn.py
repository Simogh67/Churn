# library doc string
"""
this module contains tests to apply unit tests on the churn_library.py.

Author:Simogh67
Creation Date:20.03.2024
"""

import os
import logging
import pytest
import churn_library as cls

logging.basicConfig(
    filename='logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

"""
The test functions use the return of path() as an argument
"""


@pytest.fixture(scope="module")
def path():
    return './data/bank_data.csv'


def test_import(path):
    '''
    test data import
    '''
    try:
        dataframe = cls.import_data(path)
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(path):
    """
    test perform eda function
    """
    # test if function perform_eda works
    dataframe = cls.import_data(path)
    try:
        cls.perform_eda(dataframe)
        logging.info('EDA function works')
    except KeyError as err:
        logging.error('EDA function failed')
        raise err
    # test if churn_hist.png is created
    try:
        assert os.path.isfile("./images/eda/churn_hist.png")
        logging.info('File %s was not found', 'churn_hist.png')
    except AssertionError as err:
        logging.error('File %s was not found', 'churn_hist.png')
        raise err

    # test if customer_age_distribution.png is created
    try:
        assert os.path.isfile("./images/eda/customer_age_distribution.png")
        logging.info('File %s was not found', 'customer_age_distribution.png')
    except AssertionError as err:
        logging.error('File %s was not found', 'customer_age_distribution.png')
        raise err

    # test if marital_status_distribution.png is created
    try:
        assert os.path.isfile("./images/eda/marital_status_distribution.png")
        logging.info(
            'File %s was not found',
            'marital_status_distribution.png')
    except AssertionError as err:
        logging.error(
            'File %s was not found',
            'marital_status_distribution.png')
        raise err

    # test if total_transaction_distribution.png is created
    try:
        assert os.path.isfile(
            "./images/eda/total_transaction_distribution.png")
        logging.info(
            'File %s was not found',
            'total_transaction_distribution.png')
    except AssertionError as err:
        logging.error(
            'File %s was not found',
            'total_transaction_distribution.png')
        raise err

    # test if Heatmap.png is created
    try:
        assert os.path.isfile("./images/eda/Heatmap.png")
        logging.info('File %s was not found', 'Heatmap.png')
    except AssertionError as err:
        logging.error('File %s was not found', 'Heatmap.png')
        raise err


def test_encoder_helper(path):
    '''
    test encoder helper
    '''
    # test if encoder_helper generates proper column names
    category_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']
    dataframe = cls.import_data(path)
    dataframe = cls.encoder_helper(dataframe, category_lst, 'Churn')
    try:
        for name in category_lst:
            assert name in dataframe.columns
            logging.info('the column names are chenged correctly')
    except AssertionError as err:
        logging.error('the column names are not correct')
        raise err


def test_perform_feature_engineering(path):
    '''
    test perform_feature_engineering
    '''
    dataframe = cls.import_data(path)
    train_data, test_data, y_train, y_test = cls.perform_feature_engineering(
        dataframe, 'Churn')
    # test if training and  test are generated correctly
    try:
        assert train_data.shape[0] > 0
        assert test_data.shape[0] > 0
        assert train_data.shape[1] > 0
        assert test_data.shape[1] > 0
        logging.info('training and test data are generated correctly')
    except AssertionError as err:
        logging.error(
            'Fail:training and test data are not generated correctly')
        raise err
    # test if targets are generated correctly
    try:
        assert len(y_train) > 0
        assert len(y_test) > 0
        logging.info('targets are generated correctly')
    except AssertionError as err:
        logging.error('Fail: targets are generated correctly')
        raise err


def test_train_models():
    '''
    test train_models
    '''
    # test if roc_curve_result.png is created
    try:
        assert os.path.isfile("./images/results/roc_curve_result.png")
        logging.info('File %s was not found',
                     './images/results/roc_curve_result.png')
    except AssertionError as err:
        logging.error(
            'File %s was not found',
            './images/results/roc_curve_result.png')
        raise err
    # test if rfc_model.pkl is created
    try:
        assert os.path.isfile("./models/rfc_model.pkl")
        logging.info('File %s was not found', './models/rfc_model.pkl')
    except AssertionError as err:
        logging.error('File %s was not found', './models/rfc_model.pkl')
        raise err
    # test if logistic_model.pkl is created
    try:
        assert os.path.isfile("./models/rfc_model.pkl")
        logging.info('File %s was not found', './models/rfc_model.pkl')
    except AssertionError as err:
        logging.error('File %s was not found', './models/logistic_model.pkl')
        raise err
    # test if feature_importances.png is created
    try:
        assert os.path.isfile("./images/results/feature_importances.png")
        logging.info('File %s was not found',
                     './images/results/feature_importances.png')
    except AssertionError as err:
        logging.error('File %s was not found',
                      './images/results/feature_importances.png')
        raise err


if __name__ == "__main__":
    test_import(path)
    test_eda(path)
    test_encoder_helper(path)
    test_perform_feature_engineering(path)
    test_train_models(path)
