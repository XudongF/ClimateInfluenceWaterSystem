import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats



def cleaned_data():
    # Get break data
    break_original = gpd.read_file('../data/break_records/break_records.shp',
                                   driver='FileGDB')
    break_original = break_original.to_crs(epsg=2026)

    # add soil type to break point
    soil_type = gpd.read_file('../data/OH035/spatial/soilmu_a_oh035.shp')
    soil_type = soil_type.to_crs(epsg=2026)
    soil_type['soil_ID'] = np.arange(1, len(soil_type) + 1)
    break_original = gpd.sjoin(break_original, soil_type, op='intersects')

    break_record = pd.DataFrame(break_original)
    break_record = break_record[(break_record['NEAR_DIST'] >= 0) & (break_record['NEAR_DIST'] <= 150)]

    # Get pipe information
    pipe_info = gpd.read_file('../data/CWRU_Pipe_Data.gdb', driver='FileGDB', layer=0)

    pipe_info = pipe_info.to_crs(epsg=2026)
    pipe_info_soil = gpd.overlay(df1=pipe_info, df2=soil_type, how='intersection')

    pipe_data = pd.DataFrame(pipe_info_soil)
    pipe_data.index = np.arange(1, len(pipe_data) + 1)


    pipe_data['INSTALLDATE'] = pd.to_datetime(pipe_data['INSTALLDATE'], errors='coerce', yearfirst=True).dt.tz_convert(
        'UTC')

    # Data cleanning for both breaks and pipes
    break_record['used_time'] = break_record['DATE_INITI'].combine_first(break_record['DATE_COMPL'])
    break_record['used_time'] = pd.to_datetime(break_record['used_time'], errors='coerce',
                                               yearfirst=True).dt.tz_localize('UTC')

    # these break records don't have breaking time information. We reomve it
    break_record.dropna(subset=['used_time'], inplace=True)
    # we remove columns whose missing data is larger than 70%
    pipe_data = pipe_data[pipe_data.columns[pipe_data.isnull().mean() < 0.7]]

    pipe_data = pipe_data[pipe_data.ASBUILTLENGTH >10]

    pipe_data = pipe_data[(np.abs(stats.zscore(pipe_data['ASBUILTLENGTH'])) < 3)]
    pipe_data = pipe_data[pipe_data.INSTALLDATE < '2020-1-1']
    pipe_data.dropna(subset=['INSTALLDATE', 'ASBUILTLENGTH', 'MUSYM', 'MATERIAL'], inplace=True)

    # we remove break records brefore the pipe's installation
    def get_pipe_data(s):
        if s.NEAR_FID in pipe_data.index:
            s['pipe_time'] = pipe_data.loc[s.NEAR_FID, 'INSTALLDATE']
            s['pipe_length'] = pipe_data.loc[s.NEAR_FID, 'ASBUILTLENGTH']
            s['MATERIAL'] = pipe_data.loc[s.NEAR_FID, 'MATERIAL']
        else:
            s['pipe_time'] = np.nan
            s['pipe_length'] = np.nan
        return s

    break_record = break_record.apply(get_pipe_data, axis=1)
    break_record = break_record.dropna(subset=['pipe_time', 'MATERIAL','MUSYM', 'pipe_length'])
    break_record = break_record[break_record['used_time'] > break_record['pipe_time']]
    return break_record, pipe_data


def get_pipe_number(s):
    return len(pipe_record[pipe_record['INSTALLDATE'].dt.year < s.year])


if __name__ == '__main__':
    break_record, pipe_record = cleaned_data()
    break_record = break_record[break_record.used_time.dt.year < 2020]

    break_record.to_pickle('../data/BreakRecord.pkl')
    pipe_record.to_pickle('../data/PipeRecord.pkl')

    year_break = break_record.groupby(break_record.used_time.dt.year.rename('year')).size().reset_index()
    year_break.loc[:, 'pipe_number'] = year_break.apply(get_pipe_number, axis=1)

    with plt.style.context(['science', 'no-latex']):
        fig = plt.figure(figsize=(5, 3))
        ax1 = fig.add_subplot(111)
        ax1.plot(year_break['year'].values, year_break[0].values, label='Annual break')
        ax1.set_ylabel('Break number', color='black')
        ax1.set_xlabel('Year')
        ax2 = ax1.twinx()
        ax2.plot(year_break['year'].values, year_break['pipe_number'].values, c='green',ls='--', label='Pipe number')
        ax2.set_ylabel('Pipe number', color='black')
        for tl in ax2.get_yticklabels():
            tl.set_color('blue')

        for ax in [ax1, ax2]:
            ax.tick_params(direction='in',
                           bottom=True, top=True)
        fig.legend(loc="upper left", bbox_to_anchor=(0, 1), bbox_transform=ax1.transAxes)
        plt.tight_layout()
        plt.savefig('../results/MonthlyPrediction/BreakNumber.tiff')
        plt.show()
