"""
tempdata.py
2015 Brandon Mechtley
Reuman Lab, Kansas Biological Survey

Module for loading climate model surface temperature data.

TODO: I am currently working on refactoring this so that each dataset contain
    its own list of locations and own list of years so that they can be
    disjoint. For some data, e.g. ERA-Interim, it may be best to query an API,
    if possible, saving the resulting dataset as a cached copy. The same could
    be said, eventually, about CMIP5/CORDEX data.
"""

import os
import json
import os.path
import cPickle
import urllib2
import urlparse
import collections

import numpy as np
import matplotlib.cm

import ensembles


def load_met(data_path, force=False):
    """
        Load UK weather station data from downloaded MET datasets.

        :return: Dataset dict for UK weather stations. See load_all_data for
            explanation of data in each key.
    """

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    met_path = os.path.join(data_path, 'met.npz')

    if force or not os.path.exists(met_path):
        print 'Loading MET data.'

        meturi = 'http://www.metoffice.gov.uk/pub/data/weather/uk/climate/'
        stations = json.loads(urllib2.urlopen(
            urlparse.urljoin(meturi, 'historic/historic.json')
        ).read())

        stations.update(stations.get('open', {}))
        stations.update(stations.get('closed', {}))
        del stations['open']
        del stations['closed']

        for station_name, station in stations.iteritems():
            datauri = urlparse.urljoin(
                meturi, 'stationdata/%s' % station['url']
            )
            text = urllib2.urlopen(datauri).read().splitlines()

            # MET data has a really bizarre format.
            data = np.array([
                [
                    float(stripped) if len(stripped) else np.nan
                    for stripped in [
                        token.rstrip('*-')
                        for token in line.split()[:4]
                    ]
                ]
                for line in text[7:] if 'Site' not in line
            ])

            # Read in data, get winter averages.
            years, months, tmax, tmin = [data[:, i] for i in range(4)]
            years = np.array(years, dtype=int)
            tavg = (tmax + tmin) / 2

            station['years'] = np.sort(np.unique(years))
            station['twinter'] = np.array([
                np.mean(np.concatenate([
                    year_tavg[year_months == m] for m in [1, 2, 3, 12]
                ]))
                for year_tavg, year_months in [
                    (tavg[years == y], months[years == y])
                    for y in station['years']
                ]
            ])

        # Union of all years.
        all_years = sorted(set().union(*[
            set(s['years']) for s in stations.itervalues()
        ]))

        met = dict(
            years=all_years,
            names=stations.keys(),
            locations=np.array(
                [[s['lat'], s['lon']] for s in stations.values()]
            ),
            data=np.array([
                [
                    s['twinter'][s['years'] == year]
                    if year in s['years'] else np.nan
                    for year in all_years
                ]
                for s in stations.values()
            ])
        )

        np.savez(met_path, **met)
    else:
        met = np.load(met_path)

    return met


def load_all_data(esgf_path, data_path):
    """
    Loads everything (MET UK weather station data, ERA-Interim re-analysis,
    CMIP5 data, CORDEX data) for the locations specified by MET stations and
    spits it out in a tuple.

    TODO: Make this a little more configurable, rather than hardcoding dataset
        paths / properties.

    :return: (locations, datasets, experiments, singles, projects)
        locations (np.array): (L, 2) array of L latitude/longitude pairs. These
            will be in the same order as all the datasets.
        datasets: flat dictionary where each key is either 'met', 'eraint', or
            '[project]-[experiment]'.
        experiments: dictionary with a key for each possible cmip5 / cordex
            experiment (i.e. evaluation, rcp26, rcp45, rcp85). Does not contain
            singles.
        singles: flat dictionary only containing single-element ensembles (i.e.
            met, eraint).
        projects: dictionary containing information about the projects (CMIP5,
            CORDEX, ana4MIPs). It does not include any data, but is rather used
            as a template to initialize the ensembles.

        Each dataset contained in locations, datasets, experiments, and singles
        is stored as a dictionary with keys:
            data: (T, L, N) array of average winter temperatures for T years,
                L locations, and N ensemble members (e.g. different climate
                models, parameter tunings, etc.). For 'met' and 'eraint', N=1.
            years: (T,) array of years corresponding to each row of the data.
            color: matplotlib plotting color.
            plotargs: dictionary of other arguments for plotting.
    """

    # Single sources, e.g. weather station data, ERA-Interim re-analysis.
    singles = collections.OrderedDict()
    singles['met'] = dict(
        name='MET',
        colors=['k'],
        type='single',
        plotargs=dict(lw=2)
    )
    singles['met'].update(load_met(data_path))

    # TODO: Load ERA-Interim as modelset from ana4MIPs.

    # Datasets will contain all datasets, including each individual ensember
    # member from the "modelsets."
    datasets = collections.OrderedDict()

    # Sources with several sub-sources, i.e. ensemble members (different models,
    # forcing parameters, etc.)
    projects = collections.OrderedDict()
    projects['cmip5'] = dict(
        name='CMIP5',
        color='cyan',
        type='ensemble',
        plotargs=dict(alpha=0.25),
        colormap=matplotlib.cm.winter,
        path=os.path.join(esgf_path, 'CMIP5'),
        experiments=['rcp26', 'rcp45', 'rcp85']
    )

    projects['cordex'] = dict(
        name='CORDEX',
        color='orange',
        type='ensemble',
        plotargs=dict(alpha=0.25),
        colormap=matplotlib.cm.autumn,
        path=os.path.join(esgf_path, 'CORDEX'),
        experiments=['evaluation', 'rcp26', 'rcp45', 'rcp85']
    )

    projects['ana4mips'] = dict(
        name='ana4MIPs',
        color='red',
        type='ensemble',
        colormap=matplotlib.cm.Reds,
        path=os.path.join(esgf_path, 'ana4MIPs'),
        experiments=['ERA-Interim']
    )

    # "experiments" indexes all the sources by experiment (evaluation, rcp26,
    # rcp45, rcp85, etc.)
    experiments = dict()
    for project in projects.itervalues():
        for experiment in project['experiments']:
            experiments[experiment] = collections.OrderedDict()

    # Add each ensemble member to the full datasets dict and the appropriate
    # experiment dict.
    for project_name, project in projects.iteritems():
        # Interpolate the data from downloaded ESGF data if it hasn't already
        # been done.
        ensembles.interpolate_variable(
            esgf_path,
            data_path,
            project['name'],
            singles['met']['locations'],
            project['experiments'],
            'tas'
        )

        # Load winter averages, computing them or loading them from disk.
        winter = ensembles.load_winteravg(
            project['path'], project['experiments']
        )

        # For each experiment in the project, sort the models according to
        # sum of squared differences from the minimum and add it to the
        # flattened dictionaries of experiments/datasets.
        for experiment in project['experiments']:
            ensemble = dict(
                type='ensemble',
                name=project['name'],
                color=project['color'],
                colormap=project['colormap'],
                plotargs=project['plotargs'],
                data=winter[experiment]['temps'],
                years=winter[experiment]['years'],
                path=os.path.join(project['path'], experiment)
            )

            mindiffs = ensemble['data'] - np.nanmin(
                ensemble['data'], axis=(0, 2)
            )[np.newaxis, :, np.newaxis]

            ensemble['data'] = ensemble['data'][:, :, np.argsort(
                np.sum(mindiffs ** 2, axis=(0, 1))
            )]

            # Color each ensemble member according to the modelset's colormap.
            ensemble['colors'] = ensemble['colormap'](
                np.linspace(0, 1, ensemble['data'].shape[2])
            )

            set_exp_name = project_name + '-' + experiment

            datasets[set_exp_name] = ensemble
            experiments[experiment][set_exp_name] = ensemble

    for experiment in experiments:
        experiments[experiment].update(singles)

    datasets.update(singles)

    for dataset in datasets.itervalues():
        dataset['diffs'] = np.diff(dataset['data'], axis=1)

    return singles['met']['locations'], datasets, experiments, singles, projects


def main():
    locations, datasets, experiments, singles, modelsets = load_all_data(
        'ESGF', 'cache'
    )

    print locations, datasets, experiments, singles, modelsets

if __name__ == '__main__':
    main()

