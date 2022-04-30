import requests
import pandas as pd
import time
import calendar
from tqdm import tqdm

token = {'token': 'miPsQmkSKgpCNIbUcUVDRlUJmYXvwBLU'}
#
#
average_temp = []

variable = 'TMIN'
# variable = 'PRCP'
# variable = 'EVAP'
dataset_id = 'GHCND'


for year in tqdm(range(1985, 2021, 1)):
    for month in range(1, 13, 1):
        start_date = 1
        end_date = calendar.monthrange(year, month)[1]
        daily_result = requests.get("https://www.ncdc.noaa.gov/cdo-web/api/v2/data?datasetid={}&datatypeid={}"
                                     "&locationid=FIPS"
                                     ":39035&startdate"
                                     "={}-{:02d}-{:02d}&enddate={}-{:02d}-{:02d}&units=metric&limit=1000".format(
            dataset_id, variable, year, month, start_date,
            year, month, end_date), headers=token)

        daily_result = daily_result.json()
        results = pd.DataFrame(daily_result['results'])
        average_temp.append(results)
        print(pd.to_datetime([f'{year}-{month}']))
        time.sleep(0.4)


appended_data_winter = pd.concat(average_temp)

appended_data_winter.loc[:, 'date'] = pd.to_datetime(appended_data_winter['date'],errors='coerce', yearfirst=True)


appended_data_winter.set_index('date', inplace=True)
raw_data = appended_data_winter.resample('D').mean()

# raw_data.to_pickle('../data/MinimumDailyTemperature.pkl')
# raw_data.to_pickle('../data/DailyPrecipitation.pkl')
raw_data.to_pickle('../data/DailyEvaporation.pkl')