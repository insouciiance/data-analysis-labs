from datetime import timedelta
from dateutil import parser
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.tsa.api as smt
import config
from cleansing import cleanse_data
from helpers import convert_to_celsius, get_correlations, log_data_frame_info, \
    reset_pandas, \
    plot_moving_average, \
    show_column_hist, \
    plot_seasonal_decompose, \
    plot_correlation, \
    dickey_fuller_test

reset_pandas(pd)

covid_data_frame = pd.read_csv(config.DATA_PATH_COVID, index_col=["date"], parse_dates=["date"])

log_data_frame_info(covid_data_frame)

fig, ax = plt.subplots(figsize=(15, 10))

covid_data_frame[["finland_new", "sweden_new"]].plot(ax=ax)
ax.grid()
plt.show()

show_column_hist(covid_data_frame, "finland_new")
show_column_hist(covid_data_frame, "sweden_new")

plot_moving_average(covid_data_frame, "finland_new", [5, 10, 20])
plot_moving_average(covid_data_frame, "sweden_new", [5, 10, 20])

plot_seasonal_decompose(covid_data_frame, "finland_new")
plot_seasonal_decompose(covid_data_frame, "sweden_new")

plot_correlation(covid_data_frame, "finland_new")
plot_correlation(covid_data_frame, "sweden_new")

print(dickey_fuller_test(covid_data_frame, "finland_new"))
print(dickey_fuller_test(covid_data_frame, "sweden_new"))

exchange_data_frame = pd.read_csv(config.DATA_PATH_EURUAH, index_col=["Date"], parse_dates=["Date"])

log_data_frame_info(exchange_data_frame)

exchange_data_frame[["Price"]].plot(ax=ax)
ax.grid()
plt.show()

show_column_hist(exchange_data_frame, "Price")

plot_moving_average(exchange_data_frame, "Price", [5, 10, 20])

plot_seasonal_decompose(exchange_data_frame, "Price")

plot_correlation(exchange_data_frame, "Price")

print(dickey_fuller_test(exchange_data_frame, "Price"))

weather_data_frame = pd.read_csv(config.DATA_PATH_WEATHER, index_col=["DATE"], parse_dates=["DATE"])

log_data_frame_info(weather_data_frame)

cleanse_data(weather_data_frame, use_mode=True)

weather_data_frame.plot(ax=ax, subplots=True)
ax.grid()
plt.show()

weather_data_frame.loc[weather_data_frame.index[-1500:]].plot(subplots=True)
ax.grid()
plt.show()

plot_seasonal_decompose(weather_data_frame.loc[weather_data_frame.index[-1500:]].resample("W").mean(), "PRCP")

print(dickey_fuller_test(weather_data_frame, "PRCP"))

convert_to_celsius(weather_data_frame, ["TMIN", "TMAX"])

correlation_info = get_correlations(weather_data_frame, ["TMIN", "TMAX", "PRCP"])
print(correlation_info)

weather_model = smt.ARIMA(weather_data_frame["PRCP"], order=(1, 0, 2)).fit()

print(weather_model.summary())

prediction_dates = pd.date_range(parser.parse('2018-01-01') , parser.parse('2018-12-31') - timedelta(days=1), freq='d')

for date in prediction_dates:
    prediction = weather_model.predict(date)
    print(date, prediction.values[0])