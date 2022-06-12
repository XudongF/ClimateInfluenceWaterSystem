# %%
from utils import read_data
import matplotlib.pyplot as plt
from calendar import monthrange
from datetime import datetime
import pytz

plt.style.use(['science', 'no-latex', 'grid'])
plt.rcParams.update({
    'figure.figsize': (6, 4),   # specify font family here
    "font.size": 8})


def pipe_monthly_failure_rate(row, grouped_pipe, material):
    current_month = datetime(int(row.year), int(row.month), monthrange(int(row.year), int(row.month))[1], 0, 0, 0, 0,
                             pytz.UTC)

    pipe_month = grouped_pipe[grouped_pipe.INSTALLDATE <= current_month]
    # pipe_month = grouped_pipe[((
    #     current_month - grouped_pipe.INSTALLDATE).dt.days/365 <= 100) & (grouped_pipe.INSTALLDATE <= current_month)]

    pipe_length = pipe_month['ASBUILTLENGTH'].sum()
    row[f'{material}Length'] = pipe_length / 528000
    row[f'{material}FR'] = row.FailureNumber / pipe_length / \
        monthrange(int(row.year), int(row.month))[1] * 528000 * 365

    pipe_month['age'] = current_month - pipe_month.INSTALLDATE

    row[f'{material}MeanAge'] = pipe_month['age'].mean().days / 365
    row[f'{material}LowerAge'] = pipe_month['age'].quantile(0.1).days / 365
    row[f'{material}UpperAge'] = pipe_month['age'].quantile(0.9).days / 365

    return row


def failure_rate_plot(material, break_record, pipe_record):
    if material:
        break_record = break_record[break_record['MATERIAL'] == material]
        pipe_record = pipe_record[pipe_record['MATERIAL'] == material]

    failure_number = break_record.groupby(
        [break_record.used_time.dt.year, break_record.used_time.dt.month]).count().reindex(monthly_temp_sum.index,
                                                                                           fill_value=0)[['Age_label']]
    failure_number.rename(columns={'Age_label': 'FailureNumber'}, inplace=True)

    pipe_installation = pipe_record.groupby(
        [pipe_record.INSTALLDATE.dt.year, pipe_record.INSTALLDATE.dt.month]).count()[['soil_ID']]

    pipe_installation.rename(columns={'soil_ID': "PipeNumber"}, inplace=True)

    cache_index = failure_number.index
    failure_number.index = failure_number.index.set_names(['year', 'month'])
    failure_number = failure_number.reset_index()

    failure_number = failure_number.apply(
        pipe_monthly_failure_rate, args=(pipe_record, material), axis=1)

    failure_number.index = cache_index

    return failure_number, pipe_installation


def get_monthly_static(name, data):
    monthly_data = data.groupby([data.index.year, data.index.month])

    sum_value = monthly_data.mean().rename(columns={'value': f'{name}Mean'})
    std_value = monthly_data.std().rename(columns={'value': f'{name}Var'})
    accum_difference = data.diff().abs().groupby([data.index.year, data.index.month]).mean().rename(
        columns={'value': f'{name}Diff'})

    return accum_difference, sum_value, std_value


def plot_monthly_climate(data):
    name = data.columns[0]
    data.plot()
    plt.xlabel('Date')
    plt.ylabel(f'Monthly {name}')
    plt.title(f'{name}')
    plt.tight_layout()
    plt.savefig(
        f'../results/MonthlyPrediction/{name}.tiff', dpi=300, bbox_inches='tight')
    plt.show()


# %%
if __name__ == '__main__':
    break_record, min_temp, precip, pipe_record = read_data()
    pipe_record = pipe_record[(pipe_record['MATERIAL'] == 'Cast Iron') | (
        pipe_record['MATERIAL'] == 'Ductile Iron') | (pipe_record['MATERIAL'] == 'Unknown')]

    break_record = break_record[(break_record['MATERIAL'] == 'Cast Iron') | (
        break_record['MATERIAL'] == 'Ductile Iron') | (break_record['MATERIAL'] == 'Unknown')]

    # break_record = break_record[break_record['break_age'] <= 100]
# %%
    min_temp = min_temp[(min_temp.index.year < 2020) &
                        (min_temp.index.year >= 1990)]
    precip = precip[(precip.index.year < 2020) & (precip.index.year >= 1990)]

    temp_difference, monthly_temp_sum, monthly_temp_var = get_monthly_static(
        'Temp', min_temp)
    precip_difference, monthly_precip_sum, monthly_precip_var = get_monthly_static(
        'Pr', precip)

# %%
    all_failure, all_pipe = failure_rate_plot(
        material=False, break_record=break_record, pipe_record=pipe_record)

    fig, ax = plt.subplots(figsize=(4, 3))
    all_failure.groupby('year').sum().plot(y='FailureNumber', ax=ax)
    plt.xlabel('Date')
    plt.ylabel('Failure number')

    plt.show()
# %%
    # Plot the Monthly failure rates by Failures/100 miles/day
    all_failure, all_pipe = failure_rate_plot(
        'Unknown', break_record, pipe_record)
    cast_failure, cast_pipe = failure_rate_plot(
        'Cast Iron', break_record, pipe_record)
    ductile_failure, ductile_pipe = failure_rate_plot(
        'Ductile Iron', break_record, pipe_record)

    fig, ax = plt.subplots(figsize=(4, 3))
    all_failure.groupby('year').mean().plot(y=f'UnknownFR', ax=ax)
    cast_failure.groupby('year').mean().plot(y=f'Cast IronFR', ax=ax)
    ductile_failure.groupby('year').mean().plot(y=f'Ductile IronFR', ax=ax)
    plt.legend(['Unknown', 'Cast Iron', 'Ductile Iron'])
    plt.xlabel('Date')
    plt.ylabel('FailureRate(No./year/100miles)')

    plt.savefig(f'../results/MonthlyPrediction/Pipe AnnualFR.tiff',
                dpi=300, bbox_inches='tight')
    plt.show()
# %%
    fig, ax = plt.subplots(figsize=(4, 3))
    all_failure.groupby('year').mean().plot(y=f'UnknownMeanAge', ax=ax)
    cast_failure.groupby('year').mean().plot(y=f'Cast IronMeanAge', ax=ax)
    ductile_failure.groupby('year').mean().plot(
        y=f'Ductile IronMeanAge', ax=ax)
    plt.legend(['Unknown', 'Cast Iron', 'Ductile Iron'])
    plt.xlabel('Date')
    plt.ylabel('Pipe age (years)')

    plt.savefig(f'../results/MonthlyPrediction/Pipe MeanAge.tiff',
                dpi=300, bbox_inches='tight')
    plt.show()
