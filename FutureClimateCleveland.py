import os
import xarray as xr
import cftime
import matplotlib.pyplot as plt
import gc
import cartopy.crs as ccrs
# import cartopy.io.shapereader as shpreader


def generate_data(experiment_type, variable, model):
    cn_data = []

    files = [_ for _ in os.listdir(f'{model}/') if _.endswith(".nc")]

    for file_name in files:
        if (experiment_type in file_name) & (variable in file_name):
            cn_data.append(os.path.join(model, file_name))

    ds = xr.open_mfdataset(cn_data)
    if variable == 'tasmin':
        ds['tasmin'].values = ds['tasmin'].values - 273.15
        ds['tasmin'] = ds.tasmin.assign_attrs(units='C')
    elif variable == 'pr':
        ds['pr'].values = ds['pr'].values * 24 * 3600
        ds['pr'] = ds.pr.assign_attrs(units='mm')
    return ds

# def plot_global(ds, variable):
#     for year in [2020, 2040, 2060, 2080]:
#         fig = plt.figure(1, figsize=[30, 13])
#         ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
#         ax.coastlines()
#         # Pass ax as an argument when plotting. Here we assume data is in the same coordinate reference system than the projection chosen for plotting
#         # isel allows to select by indices instead of the time values
#         ds[variable].sel(time=cftime.DatetimeNoLeap(year, 1, 1)).plot.pcolormesh(ax=ax, cmap='coolwarm')
#         plt.savefig('../results/{}/{}_global_{}.tiff'.format(variable, variable, year), dpi=300, bbox_inches='tight')
#         plt.show()

def cleveland_temp(location, model):
    city, lon, lat = location
    for variable in ['tasmin', 'pr']:
        with plt.style.context(['science', 'no-latex']):
            for experiment_type in ['ssp126', 'ssp370', 'ssp585']:
                ds = generate_data(experiment_type=experiment_type, variable=variable, model = model)

                ds[variable].sel(lon=lon % 360, lat=lat, method='nearest').plot(label=experiment_type)
                df = ds[variable].sel(lon=lon % 360, lat=lat, method='nearest').to_dataframe()
                df.to_pickle('../results/{}/{}_{}_{}_data.pkl'.format(model, variable, experiment_type, city))

                del ds
                del df
                gc.collect()
            # plt.legend()
            # plt.tight_layout()
            # plt.savefig('../results/{}/{}_comparison_Cleveland.tiff'.format(model, variable), dpi=300, bbox_inches='tight')
            # plt.show()

# def global_plot():
#     # adm_shp = list(shpreader.Reader('../data/gadm36_USA_shp/gadm36_USA_1.shp').geometries())
#     ds = generate_data(experiment_type='ssp585', variable='tasmin')
#     plot_global(ds, 'tasmin')

if __name__ == '__main__':
    for location in [("Cleveland", -81.68129, 41.505493), ("Salt Lake", -111.876183, 40.758701), ("Miami", -80.191788, 25.761681), ("Phoenix",  -112.074036, 33.448376)]:
        for model in ['MIROC6', 'CanESM', 'CESM2']:
            cleveland_temp(location, model=model)
        # global_plot()