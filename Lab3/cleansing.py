from pandas import DataFrame

def cleanse_data(dataset : DataFrame, float_columns : list, negative_columns : list):
    for column in float_columns:
        normalize_floats(dataset, column)
    replace_with_average(dataset)
    invert_negative_values(dataset, negative_columns)

def normalize_floats(dataset : DataFrame, column : str):
    dataset[column] = dataset[column] \
     .str.replace(',', '.') \
     .astype(float)

def replace_with_average(dataset : DataFrame):
    # NOTE: numeric_only=None will raise TypeError in the future
    dataset.fillna(dataset.mean(numeric_only=True), inplace=True,)

def invert_negative_values(dataset : DataFrame, columns : list):
    for column in columns:
        dataset[column] = dataset[column].abs()