import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from pandas import DataFrame
from geopandas import GeoDataFrame

def show_map(geo_frame : GeoDataFrame, column : str):
    geo_frame.plot(column=column, legend=True, figsize=(10, 10), missing_kwds={
        "color": "lightgrey",
        "edgecolor": "red",
        "hatch": "///",
        "label": "Missing values"
    })
    plt.show()

def get_correlation_coefficients(
    first_dataset : DataFrame,
    second_dataset : DataFrame,
    geo_frame : GeoDataFrame):
    correlations = pd.DataFrame()
    correlations['Correlation'] = first_dataset.corrwith(second_dataset, axis=1)
    correlations['Name'] = first_dataset['Name']

    correlations_map = pd.merge(geo_frame, correlations, on=['Name'])

    return correlations, correlations_map
