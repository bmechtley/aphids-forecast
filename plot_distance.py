import itertools
import os.path

import numpy as np
import scipy.stats
import matplotlib.pyplot as pp
import geopy

import tempdata


def denan(x, y):
    xnan, ynan = np.isnan(x), np.isnan(y)
    nans = np.bitwise_or(xnan, ynan)
    return x[~nans], y[~nans]

nlocs = len(tempdata.locations)
print nlocs
distances = np.zeros([nlocs] * 2)

location_combos = list(itertools.combinations_with_replacement(range(nlocs), 2))

distances = np.array([
    geopy.distance.distance(
        tempdata.locations[l1], tempdata.locations[l2]
    /l).kilometers
    for l1, l2 in location_combos
])

sortidx = np.argsort(distances)

distances = distances[sortidx]

for experiment, datasets in tempdata.experiments.iteritems():
    pp.figure()

    for dataset in datasets.itervalues():
        data = dataset['data']

        for m in range(data.shape[2]):
            corr = np.array([
                scipy.stats.pearsonr(*denan(data[l1, :, m], data[l2, :, m]))[0]
                for l1, l2 in location_combos
            ])[sortidx]

            pp.scatter(
                distances, corr,
                color=dataset['colors'][m], s=2, **dataset.get('plotargs', {})
            )

            pp.plot(
                distances, corr,
                color=dataset['colors'][m], **dataset.get('plotargs', {})
            )


    pp.xlim(np.amin(distances), np.amax(distances))
    pp.ylim(0, 1.1)
    pp.xlabel('Distance (km)')
    pp.ylabel('r')

    pp.legend(
        [pp.Line2D(
            (0, 1), (0, 0),
            color=ds['colors'][len(ds['colors']) / 2],
            **ds.get('plotargs', {})
        ) for ds in datasets.itervalues() if len(ds['colors'])],
        [ds['name'] for ds in datasets.itervalues() if len(ds['colors'])],
        fontsize=8
    )

    plotpath = os.path.join('plots', 'distance-%s.pdf' % experiment)
    print 'Writing %s' % plotpath
    pp.savefig(plotpath)