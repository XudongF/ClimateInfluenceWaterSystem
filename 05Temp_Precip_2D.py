# %%
from utils import read_data, apply_climate, climate_shift, kriging_regression, kriging_predict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def get_failure_rate(precip_bins, precip_data, temp_bins, temp_data, break_record, pipe_record, year, step, age_thres):
    considered_breaks = break_record[
        (break_record['used_time'].dt.year >= year) & (break_record['used_time'].dt.year < year + step) & (
            break_record['break_age'] >= age_thres) & (break_record['break_age'] < age_thres + 25)]

    considered_breaks['Temp'] = pd.cut(
        considered_breaks['Temp'], temp_bins, labels=temp_bins[:-1])
    considered_breaks['Precip'] = pd.cut(
        considered_breaks['Precip'], precip_bins, labels=precip_bins[:-1])
    considered_breaks.dropna(subset=['Temp', 'Precip'], inplace=True)

    comsidered_temp = temp_data[(temp_data.index.year >= year) & (
        temp_data.index.year < year + step)]
    comsidered_pr = precip_data[(precip_data.index.year >= year) & (
        precip_data.index.year < year + step)]

    climate_days = pd.DataFrame()
    climate_days.set_index = comsidered_temp.index
    climate_days['aligned_temp'] = pd.cut(
        comsidered_temp['used_value'], temp_bins, labels=temp_bins[:-1])
    climate_days['aligned_precip'] = pd.cut(
        comsidered_pr['used_value'], precip_bins, labels=precip_bins[:-1])

    pipe_record.loc[:, 'pipe_age'] = year - pipe_record['INSTALLDATE'].dt.year

    pipe_record = pipe_record[(
        pipe_record['pipe_age'] < age_thres + 25) & (pipe_record['pipe_age'] >= age_thres)]

    pipe_length = pipe_record['ASBUILTLENGTH'].sum()
    if pipe_length < 5280:
        pipe_length = np.nan

    return considered_breaks, climate_days, pipe_length


def fitting_curve(precip, temp, failure_rates, material, age_thres):

    poly = PolynomialFeatures(degree=2)
    x_poly = poly.fit_transform(np.vstack((precip, temp)).T)
    poly_reg = LinearRegression()
    poly_reg.fit(x_poly, failure_rates)

    fitting_evaluation = poly_reg.predict(x_poly)
    r2_value = r2_score(failure_rates, fitting_evaluation)
    RMSE = np.sqrt(mean_squared_error(failure_rates, fitting_evaluation))
    print(f"{material} and {age_thres}")
    print(f"The r2 score is: {r2_value}")
    print(f"The MSE value: {RMSE}")
    return poly, poly_reg


def plot_single_year(days_data, break_data, failure_rate, precip_low, precip_up, temp_low, temp_up, year, step):
    with plt.style.context(['science', 'no-latex']):

        plt.imshow(days_data, origin='lower',
                   extent=[precip_low, precip_up, temp_low, temp_up], aspect='auto')
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('Days')
        plt.xlabel('Mean precipitation')
        plt.ylabel('Mean temperature')
        plt.title(
            "{}-{}Number of Days".format(year, year + step))
        plt.tight_layout()
        # plt.savefig(
        #     '../results/MonthlyPrediction/ExampleNumberDays_{}.tiff'.format(year), dpi=300, bbox_inches='tight')
        plt.show()
        plt.cla()
        plt.clf()

        plt.imshow(break_data, origin='lower',
                   extent=[precip_low, precip_up, temp_low, temp_up], aspect='auto')
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('Breaks')
        plt.xlabel('Mean precipitation')
        plt.ylabel('Mean temperature')
        plt.title(
            "{}-{}Number of Breaks".format(year, year + step))
        plt.tight_layout()
        plt.savefig(
            '../results/MonthlyPrediction/ExampleNumberBreaks_{}.tiff'.format(year), dpi=300, bbox_inches='tight')
        plt.show()
        plt.cla()
        plt.clf()

        failure_rate[failure_rate == 0] = np.nan
        plt.imshow(failure_rate, origin='lower',
                   extent=[precip_low, precip_up, temp_low, temp_up], aspect='auto')
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('Failure rates')
        plt.xlabel('Mean precipitation')
        plt.ylabel('Mean temperature')
        plt.title(
            "{}-{}failure_rate".format(year, year + step))
        plt.tight_layout()
        # plt.savefig(
        #     '../results/MonthlyPrediction/ExampleFailureRate_{}.tiff'.format(year), dpi=300, bbox_inches='tight')
        plt.show()
        plt.cla()
        plt.clf()


if __name__ == '__main__':
    variable = 'Mean'

    break_record, min_temp, precip, pipe_record = read_data()
    break_record = break_record[break_record.used_time.dt.year >= 1990]
    shift_time = 29
    min_temp = climate_shift(
        min_temp, shift_day=shift_time + 1, variable='Temp', variation='Mean')
    precip = climate_shift(
        precip, shift_day=shift_time + 1, variable='Pr', variation='Mean')

    for material_name in ['Cast Iron', 'Ductile Iron', 'Unknown']:
        material_break_record = break_record[break_record['MATERIAL']
                                             == material_name]
        material_pipe_record = pipe_record[pipe_record['MATERIAL']
                                           == material_name]

        material_break_record = apply_climate(
            material_break_record, climate_data=min_temp, climate_name='Temp')

        material_break_record = apply_climate(
            material_break_record, climate_data=precip, climate_name='Precip')
        # get temp-precip grid
        temp_low = math.floor(
            material_break_record['Temp'].quantile(0.05) * 100) / 100
        temp_up = math.ceil(
            material_break_record['Temp'].quantile(0.95) * 100) / 100
        precip_low = math.floor(
            material_break_record['Precip'].quantile(0.05) * 100) / 100
        precip_up = math.ceil(
            material_break_record['Precip'].quantile(0.95) * 100) / 100
        bins = 11
        temp_bins = np.linspace(temp_low, temp_up, bins)
        bins = 11
        precip_bins = np.linspace(precip_low, precip_up, bins)
        for age_thres in [0, 25, 50, 75]:

            if material_name == 'Ductile Iron' and age_thres > 30:
                break

            step = 1
            yearly_failure_rate = []
            weights = []

            example = False

            for year in range(1990, 2020, step):

                break_counts, climate_days, pipe_length = get_failure_rate(precip_bins, precip, temp_bins, min_temp, material_break_record,
                                                                           material_pipe_record, year, step, age_thres)
                yearlfaR = np.array(len(break_counts)) / \
                    np.array(pipe_length) * 528000

                break_data = np.zeros((len(temp_bins)-1, len(precip_bins)-1))
                days_data = np.zeros((len(temp_bins)-1, len(precip_bins)-1))

                for row_index in range(break_data.shape[0]):
                    for col_index in range(break_data.shape[1]):
                        break_data[row_index, col_index] = len(break_counts[(break_counts['Temp'] == temp_bins[row_index]) & (
                            break_counts['Precip'] == precip_bins[col_index])])

                        days_data[row_index, col_index] = len(climate_days[
                            (climate_days['aligned_temp'] == temp_bins[row_index]) & (
                                climate_days['aligned_precip'] == precip_bins[
                                    col_index])])

                # days_data[days_data < 2] = 0
                failure_rate = break_data / days_data / pipe_length * 528000
                failure_rate[failure_rate == np.inf] = np.nan

                if (np.sum((failure_rate > 0).any(axis=1)) < 5) or (np.sum((failure_rate > 0).any(axis=0)) < 5):
                    if example:
                        plot_single_year(days_data, break_data, failure_rate,
                                         precip_low, precip_up, temp_low, temp_up, year, step)
                    failure_rate[:] = np.nan
                    weights.append(days_data * np.nan)
                else:
                    weights.append(days_data * pipe_length)

                # failure_rate[failure_rate > 0.21] = np.nan
                yearly_failure_rate.append(failure_rate)

            mask = np.isnan(np.array(yearly_failure_rate))

            masked_FR = np.ma.MaskedArray(
                np.array(yearly_failure_rate), mask=mask)

            weighted_sum = False
            if weighted_sum:
                ma_weights = np.ma.MaskedArray(
                    np.array(weights), mask=mask)

                ma_weights = ma_weights / np.sum(ma_weights, axis=0)

                average_failure = np.multiply(masked_FR, ma_weights)
                average_failure = np.sum(average_failure, axis=0)
            else:

                average_failure = np.ma.average(masked_FR, axis=0)

            # average_failure[average_failure > 0.25] = np.nan
            upper_quartile = np.percentile(average_failure, 95)
            lower_quartile = np.percentile(average_failure, 0)
            average_failure[average_failure > upper_quartile] = np.nan
            average_failure[average_failure < lower_quartile] = np.nan

            print(
                f"material: {material_name}, age: {age_thres}, mean FR {np.mean(average_failure)}")

            with plt.style.context(['science', 'no-latex']):
                plt.imshow(average_failure, origin='lower', extent=[precip_low, precip_up, temp_low, temp_up],
                           aspect='auto')
                cbar = plt.colorbar()
                cbar.ax.set_ylabel('Failure rates')
                plt.xlabel('Mean precipitation')
                plt.ylabel('Mean temperature')
                plt.title("Average failure_rate")
                plt.tight_layout()
                plt.savefig('../results/MonthlyPrediction/test/2DFailure rate_{}_{}.tiff'.format(
                    material_name, age_thres), dpi=300, bbox_inches='tight')
                plt.show()

            if np.sum(np.isnan(average_failure)) < 20:
                x_pos, y_pos = np.nonzero(~np.isnan(average_failure))
                z_values = average_failure[x_pos, y_pos]

                X_1 = np.array([temp_bins[i] for i in x_pos])
                X_2 = np.array([precip_bins[i] for i in y_pos])

                label = r'../results/MonthlyPrediction/test/2DKriging{}{}'.format(
                    material_name, age_thres)

                poly, poly_reg = fitting_curve(
                    precip=X_2, temp=X_1, failure_rates=z_values, material=material_name, age_thres=age_thres)

                new_precip = np.linspace(0.9*precip_low, precip_up, 50)
                new_temp = np.linspace(0.9*temp_low, temp_up, 50)
                xx, yy = np.meshgrid(new_precip, new_temp)
                regressed = poly.transform(
                    np.vstack((xx.ravel(), yy.ravel())).T)
                high_res = poly_reg.predict(regressed)
                high_res = high_res.reshape((len(new_temp), len(new_precip)))
                with plt.style.context(['science', 'no-latex']):
                    plt.imshow(high_res, origin='lower', extent=[
                               precip_low, precip_up, temp_low, temp_up], aspect='auto')
                    cbar = plt.colorbar()
                    cbar.ax.set_ylabel('Failure rate')
                    plt.title('Failure rate with kriging')
                    plt.xlabel('Mean precipitation')
                    plt.ylabel('Mean temperature')
                    plt.tight_layout()
                    plt.savefig(
                        '../results/MonthlyPrediction/test/2DFailure rate_Kridged_{}_{}.tiff'.format(
                            material_name, age_thres),
                        dpi=300, bbox_inches='tight')
                    plt.show()


# %%
