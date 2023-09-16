import pandas as pd
from dotenv import load_dotenv
import os
import nasdaqdatalink
import matplotlib.pyplot as plt
load_dotenv()


NASDAQ_DATA_LINK_API_KEY = os.getenv("NASDAQ_DATA_LINK_API_KEY")
# Data for 25 years and older
colgrad_data = nasdaqdatalink.get('FRED/CGPD25O', api_key=NASDAQ_DATA_LINK_API_KEY)
noncolgrad_data = nasdaqdatalink.get('FRED/LNU04027660', api_key=NASDAQ_DATA_LINK_API_KEY)

rates = colgrad_data.merge(noncolgrad_data, how='inner', on='Date')

rates.rename(columns={'Value_x': 'colgrad_rate', 'Value_y': 'noncolgrad_rate'}, inplace=True)

plt.figure(figsize=(10, 8))
plt.plot(rates['colgrad_rate'], 'b-', label = 'Professional Degree (25 and over)')
plt.plot(rates['noncolgrad_rate'], 'r-', label = 'No Degree (25 and over)')
plt.xlabel('Date'); plt.ylabel('Unemployment Rate (%)'); 
plt.title('Unemployment Rate of those with no College Degree vs those with a Professional Degree')
plt.legend()
plt.show()
