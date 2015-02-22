"""
interpolate-cordex.py
2015 Brandon Mechtley
Reuman Lab, Kansas Biological Survey

Interpolate surface temperature (tas) data from downloaded CORDEX NetCDF files
at a list of specific locations.
"""

import os
import os.path
import distutils.dir_util

import multiprocessing

import numpy as np
import netCDF4
import scipy.interpolate


def interpolate_data(input):
    print input['id'], 'opening', input['netcdf']

    ncf = netCDF4.Dataset(input['netcdf'], 'r', format='NETCDF4')
    var = ncf.variables[input['var']][:]
    lats = ncf.variables['lat'][:]
    lons = ncf.variables['lon'][:]
    times = ncf.variables['time'][:]

    # The 44i vs 44 CORDEX data is formatted a little differently, as lat and
    # lon aren't functions computed at each grid point, but are dimensions, so
    # you just have lat = (nlat,) and lon = (nlon,) rather than each being
    # repeated (nlat, nlon) matrices.
    if lats.ndim < 2:
        print input['id'], 'Reshaping.'
        lats, lons = np.meshgrid(lats, lons)

        if var.ndim > 3:
            var = var.reshape((len(times), lats.shape[0], lats.shape[1]))

    latlons = np.array([lats.flatten(), lons.flatten()]).T

    # Could do one 3D scipy.interpolate.griddata interpolation, but it
    # appears to be much slower than just doing a 2D for each month.
    interpolated = np.zeros((len(times), len(input['locations']) + 1))

    for t, time in enumerate(times):
        var_slice = var[t, :, :]
        interpolated[t, 0] = time

        interpolated[t, 1:] = scipy.interpolate.griddata(
            latlons,
            var_slice.flatten(),
            input['locations']
        )

    print input['id'], 'writing', input['csv']
    np.savetxt(input['csv'], interpolated)


def main():
    locations = np.genfromtxt('data/locations.csv', delimiter=',')
    data_path = '../CORDEX/data'
    data_subdir = 'concatenated'
    experiments = ['evaluation', 'rcp26', 'rcp45', 'rcp85']
    variable = 'tas'

    for experiment in experiments:
        # Make the output directory for the experiment if it doesn't exist.
        distutils.dir_util.mkpath(experiment)
        experiment_path = os.path.join(data_path, experiment, data_subdir)

        netcdf_files = [
            f for f in os.listdir(experiment_path) if f.endswith('.nc')
        ]

        csv_paths = [
            os.path.join(experiment, ncfile.rstrip('.nc') + '.csv')
            for ncfile in netcdf_files
        ]

        print 'Experiment:', experiment, '(%d)' % len(netcdf_files)

        inputs = [
            {
                'csv': os.path.join(experiment, ncfile.rstrip('.nc') + '.csv'),
                'netcdf': os.path.join(experiment_path, ncfile),
                'locations': locations,
                'var': variable,
                'id': i,
            }
            for i, (ncfile, csv_path) in enumerate(zip(netcdf_files, csv_paths))
            if not os.path.exists(csv_path)
        ]

        pool = multiprocessing.Pool(3)
        pool.map(interpolate_data, inputs)

if __name__ == '__main__':
    main()
