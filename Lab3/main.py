import pandas as pd
from config import DATA_SOURCE_PATH
from cleansing import cleanse_data
from helpers import print_data_frame_info, reset_pandas, get_row_info, get_sorted_values
from plots import print_box_plots, print_histograms

data_frame = pd.read_csv(DATA_SOURCE_PATH, sep=";", encoding="cp1252")

reset_pandas(pd)

print_data_frame_info(data_frame)

cleanse_data(
    data_frame,
    float_columns=["GDP per capita", "CO2 emission", "Area"],
    negative_columns=["GDP per capita", "Area"])

print_box_plots(data_frame, ["GDP per capita", "Population", "CO2 emission", "Area"])
print_histograms(data_frame, ["GDP per capita", "Population", "CO2 emission", "Area"])

data_frame["Population density"] = data_frame["Population"] / data_frame["Area"]

max_gdp_row, max_gdp_country = get_row_info(
    data_frame,
    lambda d : d.loc[d["GDP per capita"].idxmax()],
    "Country Name")
print("Max GDP country: ", max_gdp_country)
print("Max GDP row:\n", max_gdp_row)

min_area_row, min_area_country = get_row_info(
    data_frame,
    lambda d : d.loc[d["Area"].idxmin()],
    "Country Name")
print("Min area country: ", min_area_country)
print("Min area row:\n", min_area_row)

mean_region_areas = get_row_info(
    data_frame,
    lambda d : d.groupby("Region").mean())
print("Max mean area region: ", mean_region_areas["Area"].idxmax())
print("Mean areas by region:\n", mean_region_areas["Area"])

max_world_population = get_row_info(data_frame, lambda d : d["Population"].idxmax())
print("Country with the biggest population in the world:")
print(data_frame.loc[max_world_population][["Country Name", "Population"]])

max_europe_population = get_row_info(data_frame, lambda d : d[d["Region"] == "Europe & Central Asia"]["Population"].idxmax())
print("Country with the biggest population in Europe:")
print(data_frame.loc[max_europe_population][["Country Name", "Population"]])

_, region_mean_gdp = get_row_info(data_frame, lambda d : d.groupby("Region").mean(), "GDP per capita")
_, region_median_gdp = get_row_info(data_frame, lambda d : d.groupby("Region").median(), "GDP per capita")

print("Mean GDP per capita by region:\n", region_mean_gdp)
print("Median GDP per capita by region:\n", region_median_gdp)

same_region_gdps = pd.merge(region_mean_gdp, region_median_gdp, how="inner")
print("No match" if same_region_gdps.empty else same_region_gdps)

print("Top 5 countries by GDP per capita:")
top_gdp_countries = get_sorted_values(data_frame, "GDP per capita", ascending=False)
print(top_gdp_countries)

print("Worst 5 countries by GDP per capita:")
worst_gdp_countries = get_sorted_values(data_frame, "GDP per capita")
print(worst_gdp_countries)

data_frame["CO2 emission per capita"] = data_frame["CO2 emission"] / data_frame["Population"]

print("Top 5 countries by CO2 emission per capita:")
top_emission_countries = get_sorted_values(data_frame, "CO2 emission per capita")
print(top_emission_countries)

print("Worst 5 countries by CO2 emission per capita:")
worst_emission_countries = get_sorted_values(data_frame, "CO2 emission per capita", ascending=False)
print(worst_emission_countries)
