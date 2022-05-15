from numpy import partition
import pandas as pd
import config
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from cleansing import cleanse_data
from helpers import \
    build_linear_regression, \
    create_regressions, \
    log_data_frame_info, \
    reset_pandas, \
    build_linear_regression, \
    build_polynomial_regression, \
    partition_dataframe, \
    test_regression, \
    test_regressions

wine_data_frame = pd.read_csv(config.DATA_SOURCE_PATH_WINE, sep=',', decimal='.')

reset_pandas(pd)

log_data_frame_info(wine_data_frame)

cleanse_data(wine_data_frame, [], wine_data_frame.columns.to_list())

wine_train, wine_test = train_test_split(wine_data_frame)

log_data_frame_info(wine_train)
log_data_frame_info(wine_test)

x_wine_train, y_wine_train = partition_dataframe(wine_train, "quality")
x_wine_test, y_wine_test = partition_dataframe(wine_test, "quality")

wine_linear_regression = build_linear_regression(x_wine_train, y_wine_train)
wine_polynomial_regression = build_polynomial_regression(x_wine_train, y_wine_train)

linear_mean_error, linear_r2 = test_regression(wine_linear_regression, x_wine_test, y_wine_test)
polynomial_mean_error, polynomial_r2 = test_regression(wine_polynomial_regression, x_wine_test, y_wine_test)

print("Linear regression mean error: ", linear_mean_error)
print("Linear regression r2 error: ", linear_r2)

print("Polynomial regression mean error: ", polynomial_mean_error)
print("Polynomial regression r2 error: ", polynomial_r2)

country_train = pd.read_csv(config.DATA_SOURCE_PATH_ADDITIONAL_TRAIN, encoding='windows-1251', sep=';', decimal=',')
country_test = pd.read_csv(config.DATA_SOURCE_PATH_ADDITIONAL_TEST, encoding='windows-1251', sep=';', decimal=',')

cleanse_data(country_train, [], ["Cql", "Ie", "Iec", "Is"])
cleanse_data(country_test, [], ["Cql", "Ie", "Iec", "Is"])

log_data_frame_info(country_train)
log_data_frame_info(country_test)

print(country_train.corr())

pd.plotting.scatter_matrix(country_train)

plt.show()

# choose some value to be criterion variable, this doesn't really matter here
x_country_train, y_country_train = partition_dataframe(country_train, "Cql")

x_columns_list = [["Ie", "Iec", "Is"], ["Ie", "Iec"], ["Ie", "Is"], ["Iec", "Is"], ["Is"], ["Iec"], ["Is"]]

regressions = create_regressions(x_country_train, y_country_train, build_linear_regression, x_columns_list) + \
    create_regressions(x_country_train, y_country_train, build_polynomial_regression, x_columns_list)

x_country_test, y_country_test = partition_dataframe(country_train, "Cql")

best_regression_index = test_regressions(x_country_test, y_country_test, regressions, x_columns_list)

print(best_regression_index)
