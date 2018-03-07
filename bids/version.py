from __future__ import absolute_import, division, print_function

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 5
_version_micro = ''  # use '' for first of series, number for 1 and above
_version_extra = ''
# _version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

# Description should be a one-liner:
description = "bids: interface with datasets conforming BIDS"
# Long description will go up on the pypi page
long_description = """

PyBIDS
======
PyBIDS is a Python module to interface with datasets conforming BIDS.
See BIDS paper_ and http://bids.neuroimaging.io website for more information.

.. paper_: http://www.nature.com/articles/sdata201644

License
=======
``pybids`` is licensed under the terms of the MIT license. See the file
"LICENSE" for information on the history of this software, terms & conditions
for usage, and a DISCLAIMER OF ALL WARRANTIES.

All trademarks referenced herein are property of their respective holders.

Copyright (c) 2016--, PyBIDS developers, Planet Earth

"""

NAME = "pybids"
MAINTAINER = "PyBIDS Developers"
MAINTAINER_EMAIL = "bids-discussion@googlegroups.com"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "http://github.com/INCF/pybids"
DOWNLOAD_URL = ""
LICENSE = "MIT"
AUTHOR = "PyBIDS developers"
AUTHOR_EMAIL = "http://github.com/INCF/pybids"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
# No data for now
# PACKAGE_DATA = {'bids': [pjoin('data', '*')]}
REQUIRES = ["grabbit>=0.1.1", "six", "num2words"]
EXTRAS_REQUIRE = {
    'analysis': ['numpy', 'scipy', 'pandas', 'nibabel', 'patsy'],
}
