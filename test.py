# %%
from utils import read_climate, climate_shift, read_data, kriging_predict
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def intensity_plot(precip, temp):
    xy = np.vstack([precip, temp])
    z = gaussian_kde(xy)(xy)

    with plt.style.context(['science', 'no-latex']):
        plt.scatter(precip,
                    temp, c=z)
        plt.title('Intensity')
        plt.xlabel('Mean precipitation')
        plt.ylabel('Mean temperature')
        plt.tight_layout()
        plt.show()


# %%
if __name__ == '__main__':
    climate_model = 'CanESM'
    ssp = 'ssp585'
    city = 'Cleveland'

    material = 'Cast Iron'
    future_climate = read_climate(model=climate_model, ssp=ssp, city=city)

    intensity_plot(future_climate['Pr'].values, future_climate['Temp'].values)

# %%
    _, min_temp, precip, _ = read_data()
    shift_time = 29
    min_temp = climate_shift(
        min_temp, shift_day=shift_time + 1, variable='Temp', variation='Mean')

    # add precipitation data
    shift_time = 29
    precip = climate_shift(
        precip, shift_day=shift_time + 1, variable='Pr', variation='Mean')

    min_temp = min_temp[(min_temp.index.year >= 2010) &
                        (min_temp.index.year <= 2019)]
    precip = precip[(precip.index.year >= 2010) & (precip.index.year <= 2019)]

    intensity_plot(precip.used_value.values, min_temp.used_value.values)

# %%
    age = 25
    label = r'../results/MonthlyPrediction/2DKriging{}{}'.format(material, age)
    FR_list, _ = kriging_predict(
        precip=precip.used_value.values, temp=min_temp.used_value.values, label=label, style='points')

    print(np.sum(FR_list))
# %%
    age = 75
    label = r'../results/MonthlyPrediction/2DKriging{}{}'.format(material, age)
    FR_list_2, _ = kriging_predict(
        precip=precip.used_value.values, temp=min_temp.used_value.values, label=label, style='points')
    print(np.sum(FR_list_2))
# %%
    precip_low = 0
    precip_up = 5
    temp_low = -6
    temp_up = 20
    new_precip = np.linspace(0.9*precip_low, precip_up, 50)
    new_temp = np.linspace(0.9*temp_low, temp_up, 50)
    label_1 = r'../results/MonthlyPrediction/2DKriging{}{}'.format(
        material, 25)

    label_2 = r'../results/MonthlyPrediction/2DKriging{}{}'.format(
        material, 75)

    high_res_1, ss = kriging_predict(
        new_precip, new_temp, label_1, style='grid')
    high_res_2, ss = kriging_predict(
        new_precip, new_temp, label_2, style='grid')

    with plt.style.context(['science', 'no-latex']):
        plt.imshow(high_res_1-high_res_2, origin='lower', extent=[
            precip_low, precip_up, temp_low, temp_up], aspect='auto')
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('Failure rate')
        plt.title('Failure rate with kriging')
        plt.xlabel('Mean precipitation')
        plt.ylabel('Mean temperature')
        plt.tight_layout()
        plt.show()

# %%
