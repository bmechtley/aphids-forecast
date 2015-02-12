import numpy as np
import os.path
import os

# Load MET data from Lawrence. Each row corresponds to a location from the
# met_locations, an array of latitudes and longitudes.
met_locations = np.genfromtxt('data/locations.csv', delimiter=',')

# Load the data from CMIP5 for the specified experiment. Make sure the files are
# ordered the same as the locations from the MET data.
eraint_path = 'data/knmi/eraint'
eraint_filenames = np.array([
    f for f in os.listdir(eraint_path) if f.endswith('.csv')
])

eraint_locs  = np.array([
    '%s' % t[-2]
    for t in [
        f.rstrip('.csv').split('_')
        for f in eraint_filenames
    ]
])

eraint_indices = np.array([
    np.argwhere(eraint_locs == '%.3fN' % ll[0])[0]
    for ll in met_locations
]).flatten()

eraint_filenames = eraint_filenames[eraint_indices]

eraint_data = np.array([
    np.genfromtxt(os.path.join(eraint_path, filename), delimiter=',')[:,1]
    for filename in eraint_filenames
])

eraint_data = np.vstack([1979 + np.arange(eraint_data.shape[1]), eraint_data])

csv_path = os.path.join(eraint_path, 'aphid_eraint_1979_2014.csv')
print 'Writing %s.' % csv_path
np.savetxt(csv_path, eraint_data.T, fmt='%.3f', delimiter=',')