import pandas as pd
import subprocess
import os
from pandarallel import pandarallel
from standard_precip import spi
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from pykrige.rk import Krige
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
from scipy.interpolate import interp2d


def regression_model(precip, temp, failure_rate, label):
    z_f = interp2d(precip, temp, failure_rate, 'cubic')
    if label:
        with open(f'{label}.pickle', 'wb') as handle:
            pickle.dump(z_f, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return 0, 0


def predict_value(precip, temp, label):
    with open(f'{label}.pickle', 'rb') as handle:
        model = pickle.load(handle)

    predict_result = model(precip, temp)
    return predict_result


def train_kriging(precip, temp, failure_rate, parameter):
    if parameter['method'] == 'universal':
        OK = UniversalKriging(
            precip,
            temp,
            failure_rate,
            variogram_model=parameter['variogram_model'],
            verbose=True,
            enable_plotting=False,
            drift_terms=["regional_linear"],
            nlags=parameter['nlags']
        )

    else:
        OK = OrdinaryKriging(
            precip,
            temp,
            failure_rate,
            variogram_model=parameter['variogram_model'],
            verbose=True,
            enable_plotting=False,
            nlags=parameter['nlags']
            # drift_terms=["regional_linear"]
        )
    return OK


def kriging_regression(precip, temp, failure_rate, label=None, save=True):
    param_dict = {
        "method": ["universal"],
        "variogram_model": ["linear", "power", "gaussian", "spherical"],
        "nlags": [6,   12],
        # "weight": [True, False]
    }

    estimator = GridSearchCV(
        Krige(), param_dict, verbose=True, return_train_score=True, scoring='r2')

    estimator.fit(X=np.vstack((precip, temp)).T, y=failure_rate)

    parameter = estimator.best_params_

    OK = train_kriging(precip, temp, failure_rate, parameter)

    NSE = 0
    if save:
        OK.display_variogram_model()
        plt.show()
        with open(f'{label}.pickle', 'wb') as handle:
            pickle.dump(OK, handle, protocol=pickle.HIGHEST_PROTOCOL)

        validate_value, _ = kriging_predict(
            precip, temp, label, style='points')

        NSE = 1 - np.sum(np.square(failure_rate - validate_value)) / \
            np.sum(np.square(failure_rate - np.mean(failure_rate)))

    return NSE, parameter


def kriging_predict(precip, temp, label, style):
    with open(f'{label}.pickle', 'rb') as handle:
        model = pickle.load(handle)

    predict_result, ss = model.execute(style, precip, temp)

    return predict_result, ss


def read_data(temp_address=None, precip_address=None):
    if temp_address is None:
        temp_address = '../data/MinimumDailyTemperature.pkl'
        precip_address = '../data/DailyPrecipitation.pkl'
    break_record = pd.read_pickle('../data/BreakRecord.pkl')
    pipe_record = pd.read_pickle('../data/PipeRecord.pkl')
    min_temp = pd.read_pickle(temp_address)

    break_record.drop(columns='geometry', inplace=True)
    pipe_record.drop(columns='geometry', inplace=True)

    precip = pd.read_pickle(precip_address)

    break_record['repair_length'] = break_record.apply(
        lambda row: min(row.pipe_length, 1000), axis=1)
    break_record.loc[:, 'break_age'] = break_record['used_time'] - \
        break_record['pipe_time']
    break_record['break_age'] = break_record['break_age'].dt.days / 365
    bins = [0, 50, 200]
    labels = ['Below 50 years', 'Above 50 years']
    break_record['Age_label'] = pd.cut(
        break_record['break_age'], bins=bins, labels=labels, right=False)

    return break_record, min_temp, precip, pipe_record


def get_climate(row, climate_data, climate_name):
    if climate_name:
        row[climate_name] = climate_data.loc[row.used_time].used_value
    else:
        row['Climate'] = climate_data.loc[row.used_time.tz_convert(
            None)].used_value
    return row


def apply_climate(break_record, climate_data, climate_name=None, parralization=False):
    if parralization:
        pandarallel.initialize()
        climate_data.index = climate_data.index.tz_localize('UTC')
        break_record = break_record.parallel_apply(
            get_climate, args=(climate_data, climate_name), axis=1)
    else:
        break_record = break_record.apply(
            get_climate, args=(climate_data, climate_name), axis=1)

    return break_record


def climate_shift(climate_data, shift_day, variation, variable):
    if variable == 'Temp' or variable == 'Pr':
        if variation == 'Mean':
            climate_data['used_value'] = climate_data.value.rolling(
                window=shift_day).mean()
        elif variation == 'Diff':
            climate_data['used_value'] = climate_data.diff(
            ).abs().value.rolling(window=shift_day).sum()
    elif variable == 'SPI':
        spi_daily = spi.SPI()
        climate_data = climate_data.reset_index()
        climate_data = spi_daily.calculate(climate_data, 'date', 'value', freq="D", scale=shift_day,
                                           fit_type="lmom", dist_type="gam")

        climate_data.set_axis(
            [*climate_data.columns[:-1], 'used_value'], axis=1, inplace=True)
        climate_data.set_index('date', inplace=True)

    else:
        raise TypeError("No such method")

    return climate_data


def download_data():
    name = 'wget'
    for file_name in os.listdir('../data'):
        print(file_name)
        if name in file_name:
            subprocess.call(['bash', '../data/{}'.format(file_name), '-s'])


def line_styles():
    line_style = ['--', '-.', '-', ':']
    marker_style = ['o', '+', 'x', '+']
    return line_style, marker_style


def get_pipe_length(grouped_pipe, year):

    pipe_length = grouped_pipe[(
        grouped_pipe.INSTALLDATE.dt.year <= year)]['ASBUILTLENGTH'].sum()

    return pipe_length


def func(x, a=0, b=0, c=0):

    f = a * x ** 2 + b * x + c
    # f = a * x + b
    # f = d*x**3 + a * x ** 2 + b * x + c


    return f


def read_climate(model, ssp, city):
    _, future_temp, future_precip, _ = read_data(
        temp_address=f'../results/{model}/tasmin_{ssp}_{city}_data.pkl',
        precip_address=f'../results/{model}/pr_{ssp}_{city}_data.pkl')

    future_temp = future_temp[['tasmin']].rename(columns={'tasmin': 'value'})
    future_precip = future_precip[['pr']].rename(columns={'pr': 'value'})

    # add temperature data
    shift_time = 29
    min_temp = climate_shift(
        future_temp, shift_day=shift_time + 1, variable='Temp', variation='Mean')
    min_temp.rename(columns={'value': 'OriginalTemp'}, inplace=True)
    min_temp.rename(columns={'used_value': 'Temp'}, inplace=True)
    # add precipitation data
    shift_time = 29
    precip = climate_shift(
        future_precip, shift_day=shift_time + 1, variable='Pr', variation='Mean')
    precip.rename(columns={'value': 'OriginalPr'}, inplace=True)
    precip.rename(columns={'used_value': 'Pr'}, inplace=True)

    future_climate = min_temp.join(precip)

    temp_bins = np.linspace(
        future_climate['Temp'].min(), future_climate['Temp'].max(), 30)
    pr_bins = np.linspace(
        future_climate['Pr'].min(), future_climate['Pr'].max(), 20)

    future_climate['TempRange'] = pd.cut(future_climate.Temp,
                                         bins=temp_bins, labels=temp_bins[:-1])

    future_climate['PrRange'] = pd.cut(future_climate.Pr,
                                       bins=pr_bins, labels=pr_bins[:-1])

    future_climate = future_climate[(future_climate.index.year >= 2020) & (
        future_climate.index.year < 2101)]

    return future_climate
