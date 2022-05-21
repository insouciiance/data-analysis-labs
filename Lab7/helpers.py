import matplotlib.pyplot as plt
from pandas import DataFrame
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def reset_pandas(pd):
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", None)

def log_data_frame_info(dataset : DataFrame):
    print("Data frame info:")
    dataset.info()

    print("Data format:")
    print(dataset.head(10))
    print(dataset.describe())

def show_column_hist(dataset : DataFrame, column : str):
    column_data = dataset[column]
    column_data.hist()
    plt.title(column)
    plt.show()

def plot_moving_average(dataset : DataFrame, column : str, windows : list):
    rolling_means = [(dataset[column].rolling(window=n).mean(), n) for n in windows]
    plt.title(f"Moving average")
    plt.plot(dataset[column], label="Actual values")
    for rolling_mean, window in rolling_means:
        plt.plot(rolling_mean, label=f"Rolling mean trend, n = {window}")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.show()

def plot_seasonal_decompose(dataset : DataFrame, column : str):
    decomposition = smt.seasonal_decompose(dataset[column])
    decomposition.plot()
    plt.show()

def plot_correlation(dataset : DataFrame, column : str):
    _, ax = plt.subplots(2)
    ax[0] = plot_acf(dataset[column], ax=ax[0], lags=120)
    ax[1] = plot_pacf(dataset[column], ax=ax[1], lags=120)
    plt.show()

def dickey_fuller_test(dataset : DataFrame, column : str):
    test = smt.adfuller(dataset[column], autolag="AIC")
    print("adf: ", test[0])
    print("p-value: ", test[1])
    print("Critical values: ", test[4])
    return test[0] <= test[4]["5%"] 

def convert_to_celsius(dataset : DataFrame, columns : list):
    dataset[columns] = (dataset[columns] - 32) * 5 / 9

def get_correlations(dataset : DataFrame, columns : list):
    return dataset[columns].corr()
