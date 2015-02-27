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
import os.path
import itertools

import matplotlib
matplotlib.use('agg')
matplotlib.rc('font', family='Georgia', size=10)

import numpy as np
import matplotlib.pyplot as pp

import tempdata


def flatten_list(l):
    return [item for sublist in l for item in sublist]


def main():
    diff_bins = np.linspace(-6, 6, 20)

    locations, _, experiments, singles, _ = tempdata.load_all_data('ESGF', 'cache')

    for experiment, datasets in itertools.chain(
            singles.iteritems(), experiments.iteritems()
    ):
        # Make the figure.
        pp.figure(figsize=(10, 20))

        for loc in range(len(locations)):
            # Line plots.
            pp.subplot(len(locations), 2, (loc * 2) + 1)

            for dataset in itertools.chain(
                    datasets.iteritems(), singles.iteritems()
            ):
                data, years = dataset['data'], dataset['years']
                plotargs = dataset.get('plotargs', {})

                for model in range(data.shape[2]):
                    plotdata = data[loc, :, model]
                    finite = np.isfinite(plotdata)

                    pp.plot(
                        years[finite], plotdata[finite],
                        color=dataset['colors'][model], **plotargs
                    )

            # Union of all years from every dataset.
            allyears = set([
                int(year) for years in [
                    ds['years'] for ds in datasets.itervalues()
                ] for year in years
            ])

            pp.xlim(min(allyears), max(allyears))
            pp.xlabel(r'year')
            pp.ylabel(r'$T_w (\degree C)$')
            pp.title('${0[0]}{d} N {0[1]}{d} E$'.format(
                locations[loc], d=r'\degree'
            ), x=1.1, y=1.1)

            # Distribution of first-order differences.
            ax = pp.subplot(len(locations), 2, (loc * 2) + 2)

            if loc == 0:
                ax.legend(
                    [pp.Line2D(
                        (0, 1), (0, 0),
                        color=ds['colors'][len(ds['colors']) / 2],
                        **ds.get('plotargs', {})
                    ) for ds in datasets.itervalues() if len(ds['colors'])],
                    [ds['name'] for ds in datasets.itervalues() if len(ds['colors'])],
                    fontsize=8
                )

            for dataset in datasets.itervalues():
                diffs = dataset['data']
                plotargs = dataset.get('plotargs', {})

                for model in range(diffs.shape[2]):
                    model_diffs = diffs[loc, :, model]

                    hist, edges = np.histogram(
                        model_diffs, bins=diff_bins,
                        weights=np.ones(model_diffs.shape) / len(model_diffs)
                    )

                    pp.plot(
                        edges[:-1] + (edges[1] - edges[0]) / 2.0, hist,
                        color=dataset['colors'][model], **plotargs
                    )

            pp.ylim(0, .5)
            pp.xlabel(r'$\Delta T_w (\degree C)$')
            pp.ylabel(r'$P(\Delta T_w)$')

        pp.suptitle(
            '%s mean winter temperature predictions' % experiment.upper()
        )
        pp.subplots_adjust(hspace=0.9, wspace=0.25, top=0.95, bottom=0.05)

        plot_path = os.path.join('plots', 'temps-%s.pdf' % experiment)
        print 'Writing %s.' % plot_path
        pp.savefig(plot_path)

if __name__ == '__main__':
    main()