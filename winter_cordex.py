"""
winter_cordex.py
2015 Brandon Mechtley
Reuman Lab, Kansas Biological Survey

Assuming CORDEX data has been downloaded and interpolated into separate CSV
files for each model, with columns
    [days since 1949-12-01], [temp @ location 0], [temp @ location 1] . . .,
rearrange these CSVs to have a separate CSV for each location, with columns:
    [year] [avg winter temp for model 0], [avg winter temp for model 1] . . .

Data of this form is used by plot_temps.py.
"""

import numpy as np
import itertools
import datetime
import os.path
import glob

locations = np.genfromtxt('data/locations.csv', delimiter=',')

def makedate(days):
    """Get a datetime object from the number of days past 1942-12-01."""

    date = datetime.date(1949, 12, 1) + datetime.timedelta(days)
    return datetime.date(date.year, date.month, 1)

def makemonth(days):
    """Get the month from the number of days past 1942-12-01."""

    return makedate(days).month

def makeyear(days):
    """Get the year from the number of days past 1942-12-01."""

    return makedate(days).year

def modelname(path):
    """Extract a model name from a specially formatted filename."""

    return '_'.join(path.split('_')[2:6:3])

def write_csv_verbose(path, data, **kwargs):
    """Save an array as a CSV and print that we're doing so."""

    data = np.asarray(data)

    print 'Writing %s %s.' % (path, data.shape)
    np.savetxt(path, data, delimiter=',', **kwargs)

def transform_csv(data, model_id):
    """
    For a dataset for one particular model, create an array with rows formatted
    as follows:
        [year] [month] [location id] [model id] [temp]

    Input data is one CSV per model, with rows of the format:
        [date] [temp @ location 0] [temp @ location 1] . . .

    Dates are expressed as the number of days past 1949-12-01.
    """

    return np.vstack([
        np.column_stack((
            np.vectorize(makeyear)(data[:,0]),
            np.vectorize(makemonth)(data[:,0]),
            np.full_like(data[:,0], l),
            np.full_like(data[:,0], model_id),
            data[:,l+1]
        ))
        for l in range(data.shape[1] - 1)
    ])

def main():
    data_path = os.path.join('data', 'cordex')
    winter_path = os.path.join(data_path, 'winter')

    if not os.path.exists(winter_path):
        os.makedirs(winter_path)

    for experiment in 'evaluation', 'rcp26', 'rcp45', 'rcp85':
        experiment_path = os.path.join(data_path, experiment)

        # TODO: Temporarily ignoring 44i, because the interpolation code
        # seems to be messed up. Giving really large negative numbers for
        # temperature.
        file_paths = [
            fp
            for fp in glob.glob('%s/*.csv' % experiment_path)
            if not ('44i' in fp)
        ]

        winter_experiment_path = os.path.join(winter_path, experiment)
        if not os.path.exists(winter_experiment_path):
             os.makedirs(wither_experiment_path)

        # Write model ID reference.
        write_csv_verbose(
            os.path.join(winter_experiment_path, 'models.txt'),
            [modelname(path) for path in file_paths],
            fmt='%s'
        )

        # Get everything as a giant array with rows formatted as:
        #   [year] [month] [location_id] [model_id] [temp]
        data = np.vstack([
            transform_csv(np.genfromtxt(path), model_id)
            for model_id, path in enumerate(file_paths)
        ])

        # Delete months we don't care about.
        months = data[:,1]
        data = data[np.any([months==m for m in (1, 2, 3, 12)], axis=0)]

        # Save a CSV for each location.
        for lid, location in enumerate(locations):
            # Filter out data for other locations.
            year, month, location_id, model_id, temp = np.hsplit(
                data[data[:,2]==lid], 5
            )

            unique_years = sorted(np.unique(year))
            unique_months = sorted(np.unique(month))
            unique_models = sorted(np.unique(model_id))

            # Write the average winter temperature data.
            write_csv_verbose(
                os.path.join(
                    winter_experiment_path,
                    '%.3fN_%.3fE.csv' % tuple(location)
                ),
                np.array([
                    [y] + [
                        np.mean(matched_ym_temps)
                        if len(matched_ym_temps)
                        else np.nan
                        for (m, matched_ym_temps) in [
                            (m, matched_y_temps[matched_y_models==m]) for m in unique_models
                        ]
                    ]
                    for (y, matched_y_temps, matched_y_models) in [
                        (y, temp[year==y], model_id[year==y]) for y in unique_years
                    ]
                ]),
                fmt=tuple(['%d'] + ['%f'] * len(unique_models))
            )

if __name__ == '__main__':
    main()
