from pandas import DataFrame

def reset_pandas(pd):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', None)

def print_data_frame_info(dataset : DataFrame):
    print("Data frame info:")
    dataset.info()

    print("Data format:")
    print(dataset.head(10))

def get_row_info(dataset : DataFrame, func, column_name : str = None):
    result_row = func(dataset)

    if not column_name:
        return result_row

    result_cell = result_row[column_name]
    return result_row, result_cell

def get_sorted_values(
    dataset : DataFrame,
    column_name : str,
    rows_count : int = 5,
    ascending : bool = True):
    return dataset.sort_values(by=[column_name], ascending=ascending).head(rows_count)