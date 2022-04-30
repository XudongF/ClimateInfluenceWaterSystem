# %%
from utils import read_climate, climate_shift, read_data, kriging_predict
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def intensity_plot(precip, temp):
    xy = np.vstack([precip, temp])
    z = gaussian_kde(xy)(xy)
    corr = round(np.corrcoef(precip, temp)[0, 1], 2)

    with plt.style.context(['science', 'no-latex']):
        plt.scatter(precip,
                    temp, c=z, s=1)
        plt.title('Days density')
        plt.xlabel('Mean precipitation')
        plt.ylabel('Mean temperature')
        plt.xlim(0, 5)
        plt.text(3.6, -12, f'R={corr}')
        # plt.ylim(-10, 20)
        plt.tight_layout()
        plt.savefig('../results/MonthlyPrediction/HistoricalClimate.tiff',
                    bbox_inches='tight', dpi=300)
        plt.show()


# %%
if __name__ == '__main__':
    climate_model = 'CanESM'
    ssp = 'ssp585'
    city = 'Cleveland'

    material = 'Cast Iron'
    # future_climate = read_climate(model=climate_model, ssp=ssp, city=city)

    # for year in np.arange(2020, 2100, 10):
    #     if year == 2020 or year == 2090:
    #         considered_climate = future_climate[(future_climate.index.year >= year) & (
    #             future_climate.index.year < year+10)]

    #         intensity_plot(
    #             considered_climate['Pr'].values, considered_climate['Temp'].values)

# %%
    _, min_temp, precip, _ = read_data()
    shift_time = 29
    min_temp = climate_shift(
        min_temp, shift_day=shift_time + 1, variable='Temp', variation='Mean')

    # add precipitation data
    shift_time = 29
    precip = climate_shift(
        precip, shift_day=shift_time + 1, variable='Pr', variation='Mean')

    for year in [1990]:
        considered_precip = precip[(precip.index.year >= year) &
                                   (precip.index.year < year + 30)]
        considered_min_temp = min_temp[(min_temp.index.year >= year)
                                       & (min_temp.index.year < year + 30)]

        intensity_plot(
            considered_precip.used_value.values, considered_min_temp.used_value.values)

# %%
