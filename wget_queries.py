#!/usr/bin/env python

# This creates ESGF wget scripts for a query. Stick query options in the
# options dict. Note that this only downloads wget.sh scripts, which you then
# have to run separately to actually download the NetCDF files.

import os
import sys
import stat
import json
import urllib
import distutils.dir_util

config = json.load(open(sys.argv[1], 'r'))
project_path = os.path.join(
    os.path.dirname(sys.argv[1]),
    os.path.splitext(os.path.basename(sys.argv[1]))[0]
)

for experiment in config['experiments']:
    config['query']['experiment'] = experiment

    distutils.dir_util.mkpath(os.path.join(project_path, experiment))

    url = config['baseurl'] + '&'.join([
        '%s=%s' % (k, v) for k, v in config['query'].iteritems()
    ])

    path = '%s/wget.sh' % experiment

    print url, '>', path

    urllib.urlretrieve(url, path)
    st = os.stat(path)
    os.chmod(path, st.st_mode | stat.S_IEXEC)
