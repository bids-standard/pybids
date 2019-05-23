from __future__ import absolute_import, division, print_function
import os

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

.. _paper: http://www.nature.com/articles/sdata201644

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
URL = "http://github.com/bids-standard/pybids"
DOWNLOAD_URL = ""
LICENSE = "MIT"
AUTHOR = "PyBIDS developers"
AUTHOR_EMAIL = "bids-discussion@googlegroups.com"
PLATFORMS = "OS Independent"
# No data for now
REQUIRES = ["num2words", "numpy", "scipy", "pandas>=0.23.0",
            "nibabel>=2.1", "patsy", "bids-validator", "SQLAlchemy"]
EXTRAS_REQUIRE = {
   # Just to not break compatibility with externals requiring
   # now deprecated installation schemes
   'analysis': []
}
TESTS_REQUIRE = ["pytest>=3.3.0", 'pathlib; python_version < "3.4"']


def package_files(directory):
    # from https://stackoverflow.com/questions/27664504/how-to-add-package-data-recursively-in-python-setup-py
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

extra_files = package_files('path_to/extra_files_dir')
PACKAGE_DATA = {
    'bids.layout': ['config/*.json', 'config/validator/*.json'],
    'bids.reports': ['config/*.json'],
    'bids': package_files('bids/tests/data')
}
