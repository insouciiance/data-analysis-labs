from pandas import DataFrame

def cleanse_data(dataset : DataFrame, \
    float_columns : list = [], \
    negative_columns : list = [], \
    use_mode = False):
    for column in float_columns:
        normalize_floats(dataset, column)
    if use_mode:
        replace_with_mode(dataset)
    else:
        replace_with_mean(dataset)
    invert_negative_values(dataset, negative_columns)

def normalize_floats(dataset : DataFrame, column : str):
    dataset[column] = dataset[column] \
     .str.replace(',', '.') \
     .astype(float)

def replace_with_mode(dataset : DataFrame):
    for column in dataset.columns:
        dataset[column].fillna(dataset[column].mode()[0], inplace=True)

def replace_with_mean(dataset : DataFrame):
    dataset.fillna(dataset.mean(numeric_only=True), inplace=True)

def invert_negative_values(dataset : DataFrame, columns : list):
    for column in columns:
        dataset[column] = dataset[column].abs()
