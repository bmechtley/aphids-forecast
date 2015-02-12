import matplotlib
matplotlib.use('agg')

from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as pp

locations = np.genfromtxt('data/locations-cells.csv', delimiter=',')
parallels = np.unique(locations[:,(2,3)].flatten())
meridians = np.unique(locations[:,(4,5)].flatten())

m = Basemap(
    projection='merc',
    llcrnrlat=50,
    urcrnrlat=60,
    llcrnrlon=-6,
    urcrnrlon=6,
    resolution='i',
    lat_ts=20
)

m.drawcoastlines()
m.drawparallels(parallels, labels=[True]*len(parallels))
m.drawmeridians(meridians, labels=[True]*len(meridians))
m.drawmapboundary(fill_color='white')

x, y = m(locations[:,1], locations[:,0])
m.scatter(x,y,marker='o', color='r')

pp.title('MET Weather Stations for Aphid Data', y=1.07)
pp.savefig('plots/locations-map.pdf')
