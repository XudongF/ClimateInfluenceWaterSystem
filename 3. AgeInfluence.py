
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from utils import read_data, func, line_styles


plt.style.use(['science', 'no-latex'])

plt.rcParams.update({
    "figure.figsize": (6, 4)})


def get_failure_rate(s, pipe_data, year):

    tt = pipe_data.loc[s['age']]
    if tt['ASBUILTLENGTH'] < 5280:
        s[f'failure_rate_{year}'] = np.nan
        s[f'length_{year}'] = np.nan

    else:
        s[f'failure_rate_{year}'] = s[0] / tt['ASBUILTLENGTH'] * 528000
        s[f'length_{year}'] = tt['ASBUILTLENGTH'] / 528000

    s[f'failure_number_{year}'] = s[0]

    return s


def yearly_failure_rate(break_record, pipe_record, year):
    year_break = break_record[break_record['used_time'].dt.year == year]
    pipe_record.loc[:, 'age'] = year - pipe_record.INSTALLDATE.dt.year
    annual_pipe = pipe_record[pipe_record['age'] > 0]

    failure_number = year_break.groupby(['break_age']).size().to_frame()
    pipe_length = annual_pipe.groupby(['age']).sum()
    failure_number = failure_number.reindex(pipe_length.index, fill_value=0)
    failure_number['age'] = failure_number.index

    failure_number = failure_number.apply(
        get_failure_rate, pipe_data=pipe_length, year=year, axis=1)

    failure_number.drop(columns=[0], inplace=True)

    return failure_number


def fitting_curve(bins, failure_rates):
    nan_idx = []
    for i, arr in enumerate(failure_rates):
        if ~np.isfinite(arr) or arr == 0:
            nan_idx.append(i)

    failure_ratio_used = np.delete(failure_rates, nan_idx)
    bin_edges_used = np.delete(bins, nan_idx)
    popt, pcov = curve_fit(func, bin_edges_used, failure_ratio_used)

    fitted_value = func(bins, *popt)
    return fitted_value


def FR_Age(interested_material):
    break_record, _, _, pipe_record = read_data()
    break_record.loc[:, 'break_age'] = break_record['break_age'].astype(int)

    break_record = break_record[break_record['MATERIAL']
                                == interested_material]
    pipe_record = pipe_record[pipe_record['MATERIAL'] == interested_material]
    failures = []
    for year in range(1990, 2020):
        failure = yearly_failure_rate(
            break_record, pipe_record, year)
        failures.append(failure)

    final_failure = pd.concat(failures, axis=1)

    considered_name = [f'failure_rate_{i}' for i in range(1990, 2020)]
    weighted_name = [f'length_{i}' for i in range(1990, 2020)]

    kept_idex = final_failure.index[final_failure[considered_name].isna().sum(
        1) < 28]

    failure_rate = final_failure[considered_name].loc[kept_idex]
    failure_length = final_failure[weighted_name].loc[kept_idex]

    weighted_sum = False

    if weighted_sum:

        weights = failure_length.div(failure_length.sum(axis=1), axis=0)

        averaged_failure = failure_rate.mul(np.array(weights))

        averaged_failure = averaged_failure.sum(axis=1) / 365
    else:

        averaged_failure = failure_rate.mean(axis=1, skipna=True) / 365

    averaged_failure = averaged_failure[averaged_failure.values <= 0.25]

    fitted_value = fitting_curve(
        averaged_failure.index.values, averaged_failure.values)

    return averaged_failure, fitted_value


if __name__ == '__main__':
    line_style, marker_style = line_styles()

    for count, material in enumerate(['Cast Iron', 'Ductile Iron', 'Unknown']):
        averaged, fitted = FR_Age(material)
        averaged.replace(0, np.nan, inplace=True)
        plt.scatter(averaged.index[:100], averaged.values[:100],
                    alpha=0.5, marker=marker_style[count])
        plt.plot(averaged.index[:100], fitted[:100],
                 label=material, linewidth=2, linestyle=line_style[count])

    plt.ylim(0.02, 0.15)
    plt.legend()
    plt.xlabel("Pipe age (years)")
    plt.ylabel("FR (No./day/100 miles)")
    plt.tight_layout()
    plt.savefig("../results/MonthlyPrediction/Cast-DuctileCohortAnalysis.tiff",
                dpi=300, bbox_inches='tight')
    plt.show()

    #
    # CI_averaged, CI_fitted = FR_Age('Cast Iron')
    # DI_averaged, DI_fitted = FR_Age('Ductile Iron')
    #
    # with plt.style.context(['science', 'no-latex', 'grid']):
    #     plt.figure(figsize=(6,4))
    #     plt.scatter(CI_averaged.index, CI_averaged.values, c='blue', alpha=0.6)
    #     plt.plot(CI_averaged.index, CI_fitted, label='Cast Iron', linewidth=2, c='blue')
    #     plt.scatter(DI_averaged.index, DI_averaged.values, c='red', alpha=0.6)
    #     plt.plot(DI_averaged.index, DI_fitted, label='Ductile Iron', linewidth=2, c='red')
    #     plt.legend()
    #     plt.xlabel("Pipe age")
    #     plt.ylabel("FR (No./day/100 miles)")
    #     plt.tight_layout()
    #     plt.savefig("../results/MonthlyPrediction/Cast-DuctileCohortAnalysis.tiff", dpi=300, bbox_inches='tight')
    #     plt.show()

    #


# %%
