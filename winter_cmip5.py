"""
winter_cmip5.py
2015 Brandon Mechtley
Reuman Lab, Kansas Biological Survey

Assuming CMIP5 temperature data has been downloaded and parsed into files
with paths [experiment]/[experiment]_[Lon]E_[Lat]N/[...][model#].dat, where
each file has rows of the format
    [year] [avg temp for month1] [avg temp for month2] . . .,
parse these into separate CSV files for each location with columns according to
    [year] [avg winter temp of model 0] [avg winter temp of model 1] . . .

Data of this format is used by plot_temps.py.
"""

import os
import sys
import glob
import os.path
import numpy as np

data_path = os.path.join('data', 'knmi', 'cmip5')
winter_path = os.path.join(data_path, 'winter')

for experiment in 'rcp26', 'rcp45', 'rcp85':
    experiment_path = os.path.join(data_path, experiment)

    for location in next(os.walk(experiment_path))[1]:
        filepaths = glob.glob(os.path.join(experiment_path, location, '*.dat'))

        tokens = location.split('_')
        lon = [float(t.rstrip('E')) for t in tokens if t.endswith('E')][0]
        lat = [float(t.rstrip('N')) for t in tokens if t.endswith('N')][0]

        winter_experiment_path = os.path.join(winter_path, experiment)
        if not os.path.exists(winter_experiment_path):
            os.makedirs(winter_experiment_path)

        csv_path  = os.path.join(
            winter_experiment_path, '%.3fN_%.3fE.csv' % (lat, lon)
        )

        winters = {}

        for i, filepath in enumerate(filepaths):
            winter_temps = np.loadtxt(filepath)[:,[0, 1, 2, 3, 12]]

            if winters is None:
                winters = np.zeros((len(temps) - 1, len(filepaths) + 1))
                winters[:,0] = winter_temps[1:,0]

            winter_temps[:,-1] = np.roll(winter_temps[:,-1], 1)

            winter_temps = np.vstack([
                winter_temps[1:,0],
                np.mean(winter_temps[:,1:-1], axis=1)[1:].T
            ]).T

            for row in winter_temps:
                if row[0] not in winters:
                    winters[row[0]] = np.ones((len(filepaths),)) * -1

                winters[row[0]][i] = row[1]

        wintermat = np.vstack([
            np.array([k] + list(winters[k]))
            for k in sorted(winters.keys())
        ])

        print 'Writing %s %s.' % (csv_path, str(wintermat.shape))

        np.savetxt(csv_path, wintermat, delimiter=',', fmt='%.4f')