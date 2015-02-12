"""
plot_temps.py
2015 Brandon Mechtley
Reuman Lab, Kansas Biological Survey

Plot a) average winter temperatures and b) histograms of one-year difference
in average winter temperature across a number of locations for:
    1: ERA-Interim re-analysis data.
    2: MET UK weather station data.
    3: All CMIP5 global climate model predictions.
    4: All CORDEX regional climate model predictions.

Both CMIP5 and CORDEX data should be in one CSV file per location
(data/knmi/cmip5/[experiment]/icmip5_tas_avgwinter_[...][lon]E[lat]N.csv and
data/cordex/[experiment]_[lat]N_[lon]E_winter.csv) with rows of the format:
    [year] [avg winter temp of model 0] [avg winter temp of model 1] . . .,
ERA-Interim data should be in one CSV file per location of the format
    [year] [avg winter temp],
and MET data should be in one CSV file of the format
    [year] [avg winter temp @ location 0] [avg winter temp @ location 1] . . .

Locations are stored in data/locations.csv. Note that all locations should be
in the same order.
"""

import os
import sys
import glob
import os.path
import functools
import operator
import itertools

import matplotlib
#matplotlib.use('agg')
matplotlib.rc('font', family='Georgia', size=10)

import numpy as np
import matplotlib.pyplot as pp
import matplotlib.font_manager
import matplotlib.cm

def load_models(experiment_path, locations):
    """
    Load average winter temperature data for a particular experiment and
    location, cropping off any data before the first year of interest (available
    MET data).
    """

    filepaths = np.array(glob.glob('%s/*.csv' % experiment_path))

    # First, sort all the files the same as the locations array for
    # consistency.
    file_locations  = np.array([
        '%s_%s' % (t[0], t[1])
        for t in [
            os.path.basename(f).rstrip('.csv').split('_')
            for f in filepaths
        ] if len(t) > 1
    ])

    temps = np.array([
        np.genfromtxt(filepath, delimiter=',')
        for filepath in filepaths[np.array([
            np.argwhere(file_locations == '%.3fN_%.3fE' % tuple(ll))[0]
            for ll in locations
        ]).flatten()]
    ])

    # Return a list of years and an array of temperatures of shape
    # (# locations, # years, # models)
    return temps[0,:,0], temps[:,:,1:]

def flatten_list(l):
    return [item for sublist in l for item in sublist]

diff_bins = np.linspace(-6, 6, 20)

# Load MET weather station average winter temperature data. Each row
# corresponds to a location from "locations," an array of latitudes and
# longitudes.
locations = np.genfromtxt('data/locations.csv', delimiter=',')
first_year = 1976

met = dict(
    name='MET',
    data=np.genfromtxt('data/met/aphid_met_1976_2010.csv', delimiter=','),
    colors=['k'],
    type='single'
)
met['data'] = np.atleast_3d(met['data'])
met['years'] = first_year + np.arange(met['data'].shape[1])
met['diffs'] = np.diff(met['data'], axis=1)

# Load ERA-Interim re-analysis.
eraint = dict(
    name='ERA-Interim',
    data=np.genfromtxt(
        'data/knmi/eraint/aphid_eraint_1979_2014.csv', delimiter=','
    ),
    colors=['m'],
    type='single'
)
eraint['years'] = eraint['data'][:,0] + 1
eraint['data'] = np.atleast_3d(eraint['data'][:,1:].T - 273.15)
eraint['diffs'] = np.diff(eraint['data'], axis=1)

# Data from CMIP5/CORDEX.
cmip5 = dict(
    name='CMIP5',
    path='data/knmi/cmip5/winter/',
    type='ensemble',
    color='cyan',
    colormap=matplotlib.cm.winter,
    plotargs=dict(alpha=0.25)
)

cordex = dict(
    name='EURO-CORDEX',
    path='data/cordex/winter/',
    type='ensemble',
    color='orange',
    colormap=matplotlib.cm.autumn,
    plotargs=dict(alpha=0.25)
)

datasets = [cmip5, cordex, eraint, met]

for experiment in 'evaluation', 'rcp26', 'rcp45', 'rcp85':
    # Load all the data. CMIP5/CORDEX data are in K, so convert to C. Also make
    # sure locations are in the same order as the sources.
    for dataset in datasets:
        if dataset['type'] == 'ensemble':
            experiment_path = os.path.join(dataset['path'], experiment)

            if os.path.exists(experiment_path):
                dataset['years'], dataset['data'] = load_models(
                    experiment_path, locations
                )
                dataset['data'] -= 273.15

                # Sort the models by squared sum of distances from the minimum
                # temperature. This helps make a nice gradient in the plots.
                dataset['data'] = dataset['data'][:,:,np.argsort(
                    np.sum(
                        (
                            dataset['data'] - np.nanmin(
                                dataset['data'], axis=(0,2)
                            )[np.newaxis,:,np.newaxis] ** 2
                        ),
                        axis = (0, 1)
                    )
                )]
            else:
                dataset['data'] = np.empty((0,0,0))
                dataset['years'] = []

            dataset['diffs'] = np.diff(dataset['data'], axis=1)
            dataset['colors'] = dataset['colormap'](
                np.linspace(0, 1, dataset['data'].shape[2])
            )

    ## Make the figure.
    pp.figure(figsize=(10, 20))

    for loc in range(len(locations)):
        # Line plots.
        pp.subplot(len(locations), 2, (loc * 2) + 1)

        for dataset in datasets:
            for model in range(dataset['data'].shape[2]):
                plotdata = dataset['data'][loc,:,model]
                finite = np.isfinite(plotdata)
                plotdata = plotdata[finite]
                plotyears = dataset['years'][finite]
                pp.plot(
                    plotyears,
                    plotdata,
                    color=dataset['colors'][model],
                    **dataset.get('plotargs', {})
                )

        allyears = set([
            int(year) for years in [
                ds['years'] for ds in datasets
            ] for year in years
        ])

        pp.xlim(min(allyears), max(allyears))
        pp.xlabel(r'year')
        pp.ylabel(r'$T_w (\degree C)$')
        pp.title(
            r'$%s\degree N,\,%s\degree E$' % tuple(locations[loc]), x=1.1, y=1.1
        )

        # Distribution of first-order differences.
        ax = pp.subplot(len(locations), 2, (loc * 2) + 2)

        if loc == 0:
            ax.legend(
                [pp.Line2D(
                    (0, 1), (0, 0),
                    color=ds['colors'][len(ds['colors']) / 2],
                    **ds.get('plotargs', {})
                ) for ds in datasets if len(ds['colors'])],
                [ds['name'] for ds in datasets if len(ds['colors'])],
                fontsize=8
            )

        for dataset in datasets:
            for model in range(dataset['diffs'].shape[2]):
                diffs = dataset['diffs'][loc, :, model]

                hist, edges = np.histogram(
                    diffs, bins=diff_bins,
                    weights=np.ones(diffs.shape) / len(diffs)
                )

                pp.plot(
                    edges[:-1] + (edges[1] - edges[0]) / 2.0, hist,
                    color=dataset['colors'][model],
                    **dataset.get('plotargs', {})
                )

        pp.ylim(0, .5)
        pp.xlabel(r'$\Delta T_w (\degree C)$')
        pp.ylabel(r'$P(\Delta T_w)$')

    pp.suptitle('%s mean winter temperature predictions' % experiment.upper())
    pp.subplots_adjust(hspace=0.9, wspace=0.25, top=0.95, bottom=0.05)

    plot_path = experiment + '.pdf'
    print 'Writing %s.' % plot_path
    pp.show()
    #pp.savefig(plot_path)

