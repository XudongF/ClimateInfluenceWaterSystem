# %%
import numpy as np
import pandas as pd
from utils import read_data, kriging_predict, line_styles, read_climate
import matplotlib.pyplot as plt

plt.style.use(['science', 'no-latex'])

plt.rcParams.update({
    "figure.figsize": (4, 3)})


def update_inventory(pipe_record, year, strategy):
    if strategy == 'ductile':
        pipe_record['age'] = year - pipe_record.INSTALLDATE
        pipe_record['MATERIAL'] = np.where(
            pipe_record.age < 100, pipe_record.MATERIAL, 'Ductile Iron')
        pipe_record['INSTALLDATE'] = np.where(
            pipe_record.age < 100, pipe_record.INSTALLDATE, year)

    elif strategy == 'cast':
        pipe_record['age'] = year - pipe_record.INSTALLDATE
        pipe_record['MATERIAL'] = np.where(
            pipe_record.age < 100, pipe_record.MATERIAL, 'Cast Iron')
        pipe_record['INSTALLDATE'] = np.where(
            pipe_record.age < 100, pipe_record.INSTALLDATE, year)

    elif strategy == 'update age':
        pipe_record['age'] = year - pipe_record.INSTALLDATE
        pipe_record['INSTALLDATE'] = np.where(
            pipe_record.age < 100, pipe_record.INSTALLDATE, year)

    else:
        pipe_record['age'] = 2020 - pipe_record.INSTALLDATE

    # pipe_record['INSTAL_Length'] = np.where(pipe_record.age > 100, pipe_record.ASBUILTLENGTH, 0)

    return pipe_record


def get_yearly_failure(pipe_data, climate_data, year, material, age):

    label = r'../results/MonthlyPrediction/2DKriging{}{}'.format(material, age)

    pipe_data = pipe_data[(pipe_data['age'] >= age) &
                          (pipe_data['age'] < age + 25)]
    count = len(pipe_data)

    climate_data = climate_data[climate_data.index.year == year]

    FR_list, _ = kriging_predict(
        precip=climate_data['Pr'].values, temp=climate_data['Temp'].values, label=label, style='points')

    FR = FR_list.data
    FR[FR < 0] = 0

    FR = np.nansum(FR)

    PipeLength = pipe_data['ASBUILTLENGTH'].sum() / 528000
    FN = FR * PipeLength

    return FR, PipeLength, FN, count


def LCC_analysis(pipe_record, climate_model, ssp, city, strategy):
    future_climate = read_climate(model=climate_model, ssp=ssp, city=city)
    LCC = {}
    for year in np.arange(2020, 2101):
        pipe_record = update_inventory(pipe_record, year, strategy=strategy)
        for material in ['Ductile Iron', 'Cast Iron', 'Unknown']:

            yearly_faiure = 0
            yearly_rate = 0

            pipe_record_material = pipe_record[pipe_record['MATERIAL'] == material]
            total_number = len(pipe_record_material)

            for age in [0, 25, 50, 75]:
                FR, PipeLength, FN, count = get_yearly_failure(
                    pipe_record_material, future_climate, year, material, age)
                yearly_faiure += FN

                LCC[f'{material}PL{year}{ssp}{climate_model}{city}{age}'] = PipeLength
                LCC[f'{material}PRAge{year}{ssp}{climate_model}{city}{age}'] = FR

                yearly_rate += count / total_number * FR
            LCC[f'{material}FN{year}{ssp}{climate_model}{city}'] = yearly_faiure
            LCC[f'{material}FR{year}{ssp}{climate_model}{city}'] = yearly_rate

    return LCC


def LCC_FR_age(LCC, city):
    for material in ['Cast Iron', 'Ductile Iron', 'Unknown']:
        for age in [0, 25, 50, 75]:
            yearly_min = []
            yearly_max = []
            yearly_mean = []
            for year in np.arange(2020, 2101):
                tt = dict(
                    filter(lambda item: (material+'PRAge' + str(year) in item[0]) & (city+str(age) in item[0]), LCC.items()))

                break_value = list(tt.values())
                yearly_min.append(min(break_value))
                yearly_max.append(max(break_value))
                yearly_mean.append(np.mean(break_value))

            # plt.fill_between(np.arange(2020, 2101), yearly_min,
            #                  yearly_max, alpha=0.2)
            plt.plot(np.arange(2020, 2101), yearly_mean, label=age)

        plt.xlabel('Year')
        plt.ylabel('Failure Rate (No./year/100miles)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            f'../results/MonthlyPrediction/FRAgefluctuation{city}{material}.tiff', dpi=300, bbox_inches='tight')
        plt.show()


def LCC_yearly_plot(LCC, city):

    for material in ['Cast Iron', 'Ductile Iron', 'Unknown']:
        yearly_min = []
        yearly_max = []
        yearly_mean = []
        for year in np.arange(2020, 2101):
            year_LCC = dict(
                filter(lambda item: material + 'FN' + str(year) in item[0], LCC.items()))

            break_value = list(year_LCC.values())
            yearly_min.append(min(break_value))
            yearly_max.append(max(break_value))
            yearly_mean.append(np.mean(break_value))

        plt.fill_between(np.arange(2020, 2101), yearly_min,
                         yearly_max, alpha=0.2, label=material)
        plt.plot(np.arange(2020, 2101), yearly_mean)

    plt.xlabel('Year')
    plt.ylabel('Failure Number')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(
        f'../results/MonthlyPrediction/FNfluctuation{city}.tiff', dpi=300, bbox_inches='tight')
    plt.show()


def LCC_yearlyFR_plot(LCC, city):

    for material in ['Cast Iron', 'Ductile Iron', 'Unknown']:
        yearly_min = []
        yearly_max = []
        yearly_mean = []
        for year in np.arange(2020, 2101):
            year_LCC = dict(
                filter(lambda item: material + 'FR' + str(year) in item[0], LCC.items()))

            break_value = list(year_LCC.values())
            yearly_min.append(min(break_value))
            yearly_max.append(max(break_value))
            yearly_mean.append(np.mean(break_value))

        plt.fill_between(np.arange(2020, 2101), yearly_min,
                         yearly_max, alpha=0.2, label=material)
        plt.plot(np.arange(2020, 2101), yearly_mean)

    plt.xlabel('Year')
    plt.ylabel('Failure Rate(No./year/100miles)')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(
        f'../results/MonthlyPrediction/FRfluctuation{city}.tiff', dpi=300, bbox_inches='tight')
    plt.show()


def LCC_box_plot(LCC, city):
    yearly_values = []
    for year in np.arange(2020, 2101):
        yearly_failure = []
        for material in ['Cast Iron', 'Ductile Iron', 'Unknown']:
            year_LCC = dict(
                filter(lambda item: material+'FN'+str(year) in item[0], LCC.items()))
            yearly_failure.append(list(year_LCC.values()))

        total_failure = np.array(yearly_failure).sum(axis=0)
        yearly_values.append(total_failure)

    final_values = np.array(yearly_values)

    data = []
    for i in np.arange(0, 80, 20):
        box_data = final_values[i:i+20, :].ravel()
        data.append(box_data)
        print(f"The median value of {city} in {i} is {np.median(box_data)}")

    fig, ax = plt.subplots()
    ax.boxplot(data)
    ax.set_xlabel('Year')
    ax.set_ylabel('Failure number')
    ax.set_xticklabels(['2030', '2050', '2070', '2090'])
    plt.tight_layout()
    plt.savefig(
        f'../results/MonthlyPrediction/FN20years{city}.tiff', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':

    strategy = 'update age'
    for city in ['Cleveland', 'Miami', 'Phoenix', 'Salt Lake']:
        LCC = {}
        for model in ['MIROC6', 'CanESM', 'CESM2']:
            for ssp in ['ssp126', 'ssp370', 'ssp585']:
                _, _, _, pipe_record = read_data()
                pipe_record = pipe_record[(pipe_record['MATERIAL'] == 'Unknown') | (pipe_record['MATERIAL'] == 'Cast Iron') | (
                    pipe_record['MATERIAL'] == 'Ductile Iron')]

                pipe_record["INSTALLDATE"] = pipe_record.INSTALLDATE.dt.year

                pipe_record = pipe_record[pipe_record.INSTALLDATE >= 1920]

                LCC.update(LCC_analysis(pipe_record, climate_model=model,
                                        ssp=ssp, city=city, strategy=strategy))

        # LCC_yearly_plot(LCC, city)
        LCC_yearlyFR_plot(LCC, city)
        LCC_box_plot(LCC, city)
        # LCC_FR_age(LCC, city)
