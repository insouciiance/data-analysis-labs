from pandas import DataFrame

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