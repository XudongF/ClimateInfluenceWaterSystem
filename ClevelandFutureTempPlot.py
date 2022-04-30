from utils import read_data
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    # with plt.style.context(['science', 'no-latex']):
    #
    #     for CMIP in ['ssp126', 'ssp370', 'ssp585']:
    #         temp_address = '../results/tasmin/tasmin_{}_Cleveland_data.pkl'.format(CMIP)
    #         precip_address = '../results/pr/pr_{}_Cleveland_data.pkl'.format(CMIP)
    #         _, _, min_temp, _ = read_data(temp_address, precip_address)
    #
    #         min_temp.loc[:, 'year'] = min_temp.index.year
    #
    #         year_average_temp = min_temp.groupby('year').mean()
    #
    #         plt.plot(year_average_temp.index[:-1], year_average_temp.pr[:-1], label=CMIP)
    #     plt.legend()
    #     plt.xlabel("Year")
    #     plt.ylabel("Precipitation")
    #     # plt.savefig("../results/precipitation.tiff",dpi=300, bbox_inches='tight')
    #     plt.clf()

    break_record, min_temp, precip, pipe_record = read_data()

    break_number = break_record.groupby(break_record.used_time.dt.year).count()
    pipe_length = pipe_record.groupby(pipe_record.INSTALLDATE.dt.year).sum()
    pipe_length.loc[:, 'accumulated length'] = pipe_length['ASBUILTLENGTH'].cumsum()
    pipe_length = pipe_length.reindex(break_number.index, fill_value=np.nan).fillna(method='ffill')

    break_number.loc[:, 'break_ratio'] = break_number['break_age'].divide(pipe_length['accumulated length']) * 528000

    # #
    with plt.style.context(['science', 'no-latex', 'grid']):
        min_temp.loc[:, 'year'] = min_temp.index.year
        year_average_temp = min_temp.groupby('year').mean()
        plt.plot(year_average_temp.index[:-1], year_average_temp.value[:-1], label="History temperature")
        plt.legend()
        plt.xlabel("Year")
        plt.ylabel("Temperature")
        plt.show()

    with plt.style.context(['science', 'no-latex', 'grid']):
        plt.plot(year_average_temp.index[:-1], break_number.break_ratio, label="Annual break rate")
        plt.legend()
        plt.xlabel("Year")
        plt.ylabel("Break rate")
        plt.show()


