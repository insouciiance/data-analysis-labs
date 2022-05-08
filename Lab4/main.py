import pandas as pd
import geopandas as gpd
from additional_task.population_map.helpers import log_max_distance_info, show_image
from additional_task.gdp_map.helpers import get_correlation_coefficients, show_map
import config as config
from cleansing import cleanse_data
from helpers import log_data_frame_info, reset_pandas
from main_task.normality_tests import NormalityTestKind, log_normality_test, log_normality_tests
from main_task.helpers import log_median_significance, show_pie_diagram
import matplotlib.image as pltimg

# scrap, show and cleanse data similarly to lab3
data_frame = pd.read_csv(config.DATA_SOURCE_PATH, sep=";", encoding="cp1252")

reset_pandas(pd)

log_data_frame_info(data_frame)

cleanse_data(
    data_frame,
    float_columns=["GDP per capita", "CO2 emission", "Area"],
    negative_columns=["GDP per capita", "Area"])

log_normality_tests(
    data_frame,
    ["GDP per capita", "CO2 emission", "Area", "Population"],
    NormalityTestKind.Shapiro)

log_normality_tests(
    data_frame,
    ["GDP per capita", "CO2 emission", "Area", "Population"],
    NormalityTestKind.Pearson)

log_median_significance(data_frame, "GDP per capita")
log_median_significance(data_frame, "CO2 emission")
log_median_significance(data_frame, "Area")
log_median_significance(data_frame, "Population")

regions = pd.unique(data_frame["Region"])

for region in regions:
    region_emissions = data_frame[data_frame["Region"] == region]["CO2 emission"]
    print(f"Checking for normal distribution of CO2 emissions in: {region}")
    try:
        log_normality_test(region_emissions, NormalityTestKind.Shapiro)
        log_normality_test(region_emissions, NormalityTestKind.Pearson)
    except ValueError as e:
        #quietly skip.
        print(str(e))

show_pie_diagram(data_frame.groupby("Region").sum(), regions, "Population")

map_image = pltimg.imread(config.MAP_IMAGE_PATH)
show_image(map_image, config.CITIES_COORDS, config.CITIES_POPULATION)
log_max_distance_info(map_image, config.CITIES_COORDS, config.CITIES)

map_shape = gpd.read_file(config.MAP_SHAPE_PATH)
ukr_gdp_data_frame = pd.read_csv(config.GDP_CSV_PATH, sep=";", decimal=",", encoding="windows-1251", header=1)
ukr_dpp_data_frame = pd.read_csv(config.DPP_CSV_PATH, sep=";", decimal=",", encoding="windows-1251", header=1)

log_data_frame_info(ukr_gdp_data_frame)
log_data_frame_info(ukr_dpp_data_frame)

ukr_gdp_geo_data_frame = gpd.GeoDataFrame(pd.merge(map_shape, ukr_gdp_data_frame))
ukr_dpp_geo_data_frame = gpd.GeoDataFrame(pd.merge(map_shape, ukr_dpp_data_frame))

show_map(ukr_gdp_geo_data_frame, "2016")
show_map(ukr_dpp_geo_data_frame, "2016")

correlations, correlations_map = get_correlation_coefficients(
    ukr_dpp_data_frame,
    ukr_gdp_data_frame,
    map_shape)

log_data_frame_info(correlations)
show_map(correlations_map, "Correlation")
