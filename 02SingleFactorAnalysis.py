# %%
from unittest import result
import matplotlib.pyplot as plt
import numpy as np
from utils import read_data, apply_climate, climate_shift, line_styles, func
from scipy.optimize import curve_fit
import math
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error


plt.style.use(['science', 'no-latex'])

plt.rcParams.update({
    "figure.figsize": (4, 3)})


def fitting_curve(bins, failure_rates, material, age_thres):
    nan_idx = []
    for i, arr in enumerate(failure_rates):
        if ~np.isfinite(arr) or arr == 0:
            nan_idx.append(i)

    failure_ratio_used = np.delete(failure_rates, nan_idx)
    bin_edges_used = np.delete(bins, nan_idx)
    popt, pcov = curve_fit(func, bin_edges_used, failure_ratio_used)

    fitted_value = func(bins, *popt)

    fitting_evaluation = func(bin_edges_used, *popt)
    r2_value = r2_score(failure_ratio_used, fitting_evaluation)
    RMSE = np.sqrt(mean_squared_error(failure_ratio_used, fitting_evaluation))
    print(f"{material} and {age_thres}")
    print(f"The parameter is: {popt}")
    print(f"The r2 score is: {r2_value}")
    print(f"The MSE value: {RMSE}")
    return fitted_value


def get_failure_rate(climate, climate_bins, break_record, pipe_record, year, step, age_thres):
    considered_breaks = break_record[
        (break_record['used_time'].dt.year >= year) & (break_record['used_time'].dt.year < year + step) & (
            break_record['break_age'] >= age_thres) & (break_record['break_age'] < age_thres + 25)]

    break_numbers = pd.DataFrame()
    break_numbers.index = (climate_bins[:-1] + climate_bins[1:]) / 2
    hist_break, _ = np.histogram(
        considered_breaks['Climate'].values, climate_bins)

    break_numbers[f'FailureNumber_{year}'] = hist_break

    considered_climate = climate[(climate.index.year >= year) & (
        climate.index.year < year + step)]

    climate_days = pd.DataFrame()
    climate_days.index = (climate_bins[:-1] + climate_bins[1:]) / 2

    hist_days, _ = np.histogram(
        considered_climate['used_value'].values, climate_bins)
    climate_days[f'aligned_climate_{year}'] = hist_days

    pipe_record.loc[:, 'pipe_age'] = year - pipe_record['INSTALLDATE'].dt.year

    considered_pipes = pipe_record[(
        pipe_record['pipe_age'] < age_thres + 25) & (pipe_record['pipe_age'] >= age_thres)]

    pipe_length = considered_pipes['ASBUILTLENGTH'].sum()
    if pipe_length < 5280 * 10:
        pipe_length = np.nan

    return break_numbers, climate_days, pipe_length


def plot_agains_age(results, variable):
    line_style, marker_style = line_styles()
    for material in ['Cast Iron', 'Ductile Iron', 'Unknown']:
        for count, age_thres in enumerate([0, 25, 50, 75]):
            if f'{material}{age_thres}{variable}FR' in results:
                plt.scatter(
                    results[f'{material}{age_thres}{variable}FR'].index, results[f'{material}{age_thres}{variable}FR'].values, marker=marker_style[count], s=15)
                fitted = fitting_curve(
                    results[f'{material}{age_thres}{variable}FR'].index, results[f'{material}{age_thres}{variable}FR'].values, material, age_thres)
                plt.plot(
                    results[f'{material}{age_thres}{variable}FR'].index, fitted, linestyle=line_style[count], label=age_thres, lw=2, alpha=0.8)
            else:
                print(f"{material} at {age_thres} do not have data")

        plt.xlabel(f"{variable}")
        plt.ylabel("Failure ratio (No./day/100miles)")
        plt.legend(loc=1)
        plt.tight_layout()
        plt.savefig(
            f'../results/MonthlyPrediction/failure_ratio{material}{variable}.tiff', dpi=300, bbox_inches='tight')
        plt.show()


# %%
if __name__ == '__main__':
    results = {}
    variable = 'Mean'

    for climate_variable in ['Temp', 'Pr']:
        break_record, min_temp, precip, pipe_record = read_data()
        break_record = break_record[break_record.used_time.dt.year >= 1990]

        shift_time = 29
        if climate_variable == 'Temp':
            climate = climate_shift(
                min_temp, shift_day=shift_time + 1, variable='Temp', variation='Mean')
            break_record = apply_climate(
                break_record, climate_data=climate)
            bins = 11
        else:
            shift_time = 29
            climate = climate_shift(
                precip, shift_day=shift_time + 1, variable='Pr', variation='Mean')
            break_record = apply_climate(
                break_record, climate_data=climate)
            bins = 11

        # get temp-precip bracket
        climate_low = math.floor(
            break_record['Climate'].quantile(0.05) * 100) / 100
        climate_up = math.ceil(
            break_record['Climate'].quantile(0.95) * 100) / 100

        climate_bins = np.linspace(climate_low, climate_up, bins)

        for material_name in ['Cast Iron', 'Ductile Iron', 'Unknown']:
            break_record_material = break_record[break_record['MATERIAL']
                                                 == material_name]
            pipe_record_material = pipe_record[pipe_record['MATERIAL']
                                               == material_name]
            for age_thres in [0, 25, 50, 75]:
                if material_name == 'Ductile Iron' and age_thres > 30:
                    break

                failures = []
                weights = []

                for year in range(1990, 2020):
                    considered_breaks, climate_days, pipe_length = get_failure_rate(
                        climate, climate_bins, break_record_material, pipe_record_material, year, 1, age_thres)

                    FR = considered_breaks[f'FailureNumber_{year}'].div(
                        np.array(climate_days[f'aligned_climate_{year}'])) / pipe_length * 528000

                    weight = climate_days[f'aligned_climate_{year}'] * \
                        pipe_length / 528000
                    failures.append(FR)
                    weights.append(weight)

                final_failure = pd.concat(failures, axis=1)
                final_weights = pd.concat(weights, axis=1)
                weighted_sum = False

                if weighted_sum:
                    # final_weights = final_weights.div(
                    #     np.array(final_weights.sum(1)), axis=0)

                    final_weights = final_weights.subtract(final_weights.min(1), axis=0).div(
                        np.array(final_weights.max(1) - final_weights.min(1)), axis=0)

                    final_FR = final_failure.multiply(
                        np.array(final_weights)).sum(1)

                else:

                    final_failure = final_failure.loc[:, final_failure.gt(
                        0).sum() > 5]
                    final_FR = final_failure.mean(axis=1, skipna=True)
                results[f'{material_name}{age_thres}{climate_variable}FR'] = final_FR

        plot_agains_age(results, climate_variable)

# %%
