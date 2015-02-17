"""
tempdata.py
2015 Brandon Mechtley
Reuman Lab, Kansas Biological Survey

Module for loading climate model surface temperature data.
"""

import os
import glob
import cPickle
import collections

import numpy as np
import matplotlib.cm

import utm

# TODO: I am currently working on refactoring this so that each dataset contains
# TODO:     its own list of locations and own list of years so that they can be
# TODO:     disjoint. For some data, e.g. ERA-Interim, it may be best to query
# TODO:     an API, if possible, saving the resulting dataset as a cached copy.
# TODO:     The same could be said, eventually, about CMIP5/CORDEX data.


def load_ensemble(experiment_path, locs):
    """
    Load average winter temperature data for a particular experiment, given a
    list of locations.
    """

    filepaths = np.array(glob.glob('%s/*.csv' % experiment_path))

    # First, sort all the files the same as the locations array for
    # consistency.
    file_locations = np.array([
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
            for ll in locs
        ]).flatten()]
    ])

    # Return a list of years and an array of temperatures of shape
    # (# locations, # years, # models)
    return temps[0, :, 0], temps[:, :, 1:]


def load_met(datapath):
    """
        Load UK weather station data from downloaded MET datasets.

        :return: Dataset dict for UK weather stations. See load_all_data for
            explanation of data in each key.
    """

    filepaths = np.array(glob.glob('%s/*.txt' % datapath))

    locations = dict()

    for filepath in filepaths:
        # Read in header for lat / lon and location name.
        f = open(filepath, 'r')
        loctokens = f.readline().split(' ')
        f.close()

        # Read in data, convert from UTM to lat/lon, get winter averages.
        easting, northing = loctokens[1].rstrip('E'), loctokens[2].rstrip('N')
        lat, lon = utm.to_latlon(easting * 100, northing * 100, 30, 'U')

        data = np.genfromtxt(filepath, skip_header=7, missing_values=['---'])
        years, months, tmax, tmin, afdays, rainmm, sunhrs = data.hsplit(7)
        tavg = (tmax + tmin) / 2
        winter = np.array([1, 2, 3, 12])
        unique_years = np.sort(np.unique(years))
        twinter = np.array([
            np.mean([year_tavg[months == month] for month in winter])
            for year, year_tavg in [(y, tavg[years == y]) for y in unique_years]
        ])

        # Add it to the list of locations.
        locations[(lat, lon)] = dict(twinter=twinter, years=unique_years)

    all_years = np.sort(
        np.unique([l['years'] for l in locations.itervalues()])
    )

    return dict(
        name='MET',
        colors=['k'],
        type='single',
        plotargs=dict(lw=2),
        locations=np.array(locations.iterkeys()),
        years=all_years,
        data=np.array([
            [
                l['twinter'][year] if year in l['years'] else np.nan
                for year in all_years
            ]
            for l in locations.itervalues()
        ])
    )


def load_eraint(datapath):
    eraint = dict(
        name='ERA-Interim',
        data=np.genfromtxt(datapath, delimiter=','),
        colors=['m'],
        type='single',
        plotargs=dict(lw=2)
    )

    eraint['years'] = eraint['data'][:, 0] + 1
    eraint['data'] = np.atleast_3d(eraint['data'][:, 1:].T)

    return eraint


def load_fs_cached(picklepath, fun):
    """
    Load data produced by fun() that might optionally be cached on disk.

    :param picklepath: path to where the cached data should be.
    :param fun: function that generates the data.
    :return: the data.
    """

    if os.path.exists(picklepath):
        return cPickle.load(open(picklepath, 'r'))
    else:
        data = fun(os.path.split(picklepath)[0])
        cPickle.dump(data, open(picklepath, 'w'))
        return data


def load_all_data():
    """
    Loads everything (MET UK weather station data, ERA-Interim re-analysis,
    CMIP5 data, CORDEX data) for the locations specified in data/locations.csv
    and spits it out in a tuple.

    TODO: Make this a little more configurable, rather than hardcoding dataset
        paths / properties.

    :return: (locations, datasets, experiments, singles, modelsets)
        locations (np.array): (L, 2) array of L latitude/longitude pairs. These
            will be in the same order as all the datasets.
        datasets: flat dictionary where each key is either 'met', 'eraint', or
            '[project]-[experiment]'.
        experiments: dictionary with a key for each possible cmip5 / cordex
            experiment (i.e. evaluation, rcp26, rcp45, rcp85). Does not contain
            singles.
        singles: flat dictionary only containing single-element ensembles (i.e.
            met, eraint).
        modelsets: dictionary containing information about the modelsets. It
            does not include any data, but is rather used as a template to
            initialize the ensembles.

        Each dataset contained in locations, datasets, experiments, and singles
        is stored as a dictionary with keys:
            data: (T, L, N) array of average winter temperatures for T years,
                L locations, and N ensemble members (e.g. different climate
                models, parameter tunings, etc.). For 'met' and 'eraint', N=1.
            years: (T,) array of years corresponding to each row of the data.
            color: matplotlib plotting color.
            plotargs: dictionary of other arguments for plotting.
    """

    locations = np.genfromtxt('data/locations.csv', delimiter=',')

    # Single sources, e.g. weather station data, ERA-Interim re-analysis.
    singles = collections.OrderedDict()
    singles['met'] = load_fs_cached('data/met/met.pickle', load_met)
    singles['eraint'] = load_fs_cached(
        'data/knmi/eraint/eraint.pickle', load_eraint
    )


    # TODO: Make ensemble sets able to use load_fs_data
    
    # Datasets will contain all datasets, including each individual ensember
    # member from the "modelsets."
    datasets = collections.OrderedDict()

    # Sources with several sub-sources, i.e. ensemble members (different models,
    # forcing parameters, etc.)
    modelsets = collections.OrderedDict()
    modelsets['cmip5'] = dict(
        name='CMIP5',
        path='data/knmi/cmip5/winter/',
        type='ensemble',
        experiments=['rcp26', 'rcp45', 'rcp85'],
        color='cyan',
        colormap=matplotlib.cm.winter,
        plotargs=dict(alpha=0.25)
    )

    modelsets['cordex'] = dict(
        name='EURO-CORDEX',
        path='data/cordex/winter/',
        type='ensemble',
        experiments=['evaluation', 'rcp26', 'rcp45', 'rcp85'],
        color='orange',
        colormap=matplotlib.cm.autumn,
        plotargs=dict(alpha=0.25)
    )

    # "experiments" indexes all the sources by experiment (evaluotion, rcp26,
    # rcp45, rcp85, etc.)
    experiments = dict()
    for modelset in modelsets.itervalues():
        for experiment in modelset['experiments']:
            experiments[experiment] = collections.OrderedDict()

    # Add each ensemble member to the full datasets dict and the appropriate
    # experiment dict.
    for modelset_name, modelset in modelsets.iteritems():
        for experiment in modelset['experiments']:
            ensemble = dict(
                name=modelset['name'],
                path=os.path.join(modelset['path'], experiment),
                type='ensemble',
                color=modelset['color'],
                colormap=modelset['colormap'],
                plotargs=modelset['plotargs']
            )

            if os.path.exists(ensemble['path']):
                years, data = load_ensemble(ensemble['path'], locations)

                # Sort the models by squared sum of distances from the
                # minimum temperature. This helps make a nice gradient in
                # the plots when they're colored according to a colormap.
                mindiffs = data - np.nanmin(
                    data, axis=(0, 2)
                )[np.newaxis, :, np.newaxis]

                data = data[:, :, np.argsort(
                    np.sum(mindiffs ** 2, axis=(0, 1))
                )]
            else:
                data, years = np.empty((0, 0, 0)), []

            # Color each ensemble member according to the modelset's colormap.
            ensemble['data'], ensemble['years'] = data, years
            ensemble['colors'] = ensemble['colormap'](
                np.linspace(0, 1, data.shape[2])
            )

            set_exp_name = modelset_name + '-' + experiment

            datasets[set_exp_name] = ensemble
            experiments[experiment][set_exp_name] = ensemble

    for experiment in experiments:
        experiments[experiment].update(singles)

    datasets.update(singles)

    for dataset in datasets.itervalues():
        dataset['diffs'] = np.diff(dataset['data'], axis=1)

    return locations, datasets, experiments, singles, modelsets