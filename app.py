import pandas as pd
from dotenv import load_dotenv
import os
import nasdaqdatalink
import matplotlib.pyplot as plt
from prophet import Prophet

load_dotenv()


NASDAQ_DATA_LINK_API_KEY = os.getenv("NASDAQ_DATA_LINK_API_KEY")
# Data for 25 years and older
colgrad_data = nasdaqdatalink.get('FRED/CGPD25O', api_key=NASDAQ_DATA_LINK_API_KEY)
noncolgrad_data = nasdaqdatalink.get('FRED/LNU04027660', api_key=NASDAQ_DATA_LINK_API_KEY)

rates = colgrad_data.merge(noncolgrad_data, how='inner', on='Date')

## Non-Prophet graph
# rates.rename(columns={'Value_x': 'colgrad_rate', 'Value_y': 'noncolgrad_rate'}, inplace=True)
# plt.figure(figsize=(10, 8))
# plt.plot(rates['colgrad_rate'], 'b-', label = 'Professional Degree (25 and over)')
# plt.plot(rates['noncolgrad_rate'], 'r-', label = 'No Degree (25 and over)')
# plt.xlabel('Date'); plt.ylabel('Unemployment Rate (%)'); 
# plt.title('Unemployment Rate of those with no College Degree vs those with a Professional Degree')
# plt.legend()
# plt.show()

## Prophet graphing

colgrad_data.rename(columns={'Date': 'ds', 'Value': 'y'}, inplace=True)
colgrad_data = colgrad_data.reset_index()
colgrad_data['Date'] = pd.to_datetime(colgrad_data['Date'])
colgrad_data.rename(columns={'Date': 'ds'}, inplace=True)


a = Prophet(changepoint_prior_scale=0.15)
a.fit(colgrad_data)
colgrad_forecast = a.make_future_dataframe(periods=365)
colgrad_forecast = a.make_future_dataframe(periods=365*2, freq="D")
colgrad_forecast = a.predict(colgrad_forecast)

#############################################################

noncolgrad_data.rename(columns={'Date': 'ds', 'Value': 'y'}, inplace=True)
noncolgrad_data = noncolgrad_data.reset_index()
noncolgrad_data['Date'] = pd.to_datetime(noncolgrad_data['Date'])
noncolgrad_data.rename(columns={'Date': 'ds'}, inplace=True)

b = Prophet(changepoint_prior_scale=0.15)
b.fit(noncolgrad_data)
noncolgrad_data = b.make_future_dataframe(periods=365)
noncolgrad_forecast = b.make_future_dataframe(periods=365*2, freq="D")
noncolgrad_forecast = b.predict(noncolgrad_forecast)

colgrad_names = ['colgrad_%s' % column for column in colgrad_forecast.columns]
noncolgrad_names = ['noncolgrad_%s' % column for column in noncolgrad_forecast.columns]

merge_colgrad_forecast = colgrad_forecast.copy()
merge_noncolgrad_forecast = noncolgrad_forecast.copy()

merge_colgrad_forecast.columns = colgrad_names
merge_noncolgrad_forecast.columns = noncolgrad_names

forecast = pd.merge(merge_colgrad_forecast, merge_noncolgrad_forecast, how = 'inner', left_on = 'colgrad_ds', right_on = 'noncolgrad_ds')
forecast = forecast.rename(columns={'colgrad_ds': 'Date'}).drop('noncolgrad_ds', axis=1)

forecast_filtered = forecast[(forecast['Date'] > '1999-01-01') & (forecast['Date'] <= max(forecast['Date']))]

plt.figure(figsize=(15, 8))
plt.plot(forecast_filtered['Date'], forecast_filtered['colgrad_yhat'], 'b-', label='College Graduates')
plt.plot(forecast_filtered['Date'], forecast_filtered['noncolgrad_yhat'], 'r-', label='Non-College Graduates')
plt.xlabel('Date') 
plt.ylabel('Unemployment Rate (%))')
plt.title('Unemployment Rate of those with no College Degree vs those with a Professional Degree')
plt.xlim(pd.Timestamp('2018-01-01'), max(forecast["Date"]))
plt.ylim(0, max(forecast['noncolgrad_yhat']))
plt.margins(x=0)
plt.subplots_adjust(top=0.924)
plt.tight_layout()
plt.legend()
plt.show()
