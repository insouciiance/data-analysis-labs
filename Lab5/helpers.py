import numpy as np
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

def reset_pandas(pd):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', None)

def log_data_frame_info(dataset : DataFrame):
    print("Data frame info:")
    dataset.info()

    print("Data format:")
    print(dataset.head(10))
    print(dataset.describe())

def build_linear_regression(x_train : DataFrame, y_train : DataFrame):
    regression = LinearRegression().fit(x_train, y_train)
    return regression

def build_polynomial_regression(x_train : DataFrame, y_train : DataFrame):
    regression = make_pipeline(PolynomialFeatures(degree=4), LinearRegression()).fit(x_train, y_train)
    return regression

def test_regression(
    regression,
    x_test : DataFrame,
    y_test : DataFrame):
    predictions = regression.predict(x_test)
    mean_erorr_result = mean_squared_error(y_test, predictions)
    r2_result = r2_score(y_test, predictions)
    return mean_erorr_result, r2_result

def partition_dataframe(data_frame : DataFrame, column : str):
    except_columns = data_frame.loc[:, data_frame.columns != column]
    column = data_frame[column]
    return except_columns, column

def create_regressions(x_train : DataFrame, y_train : DataFrame, create_function, x_columns_list : list):
    return [create_function(x_train[x_columns], y_train) for x_columns in x_columns_list]

def test_regressions(x_test : DataFrame, y_test : DataFrame, regressions : list, x_columns_list : list):
    predictions = [reg.predict(x_test[x_columns_list[idx % len(x_columns_list)]]) for idx, reg in enumerate(regressions)]
    return np.sum((predictions - y_test.to_numpy()) ** 2, axis=1).argmin()
