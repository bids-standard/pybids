#!/usr/bin/env python

import os
from setuptools import setup, find_packages
PACKAGES = find_packages()

# Get version and release info, which is all stored in bids/version.py
ver_file = os.path.join('bids', 'version.py')
with open(ver_file) as f:
    exec(f.read())

opts = dict(name=NAME,
            maintainer=MAINTAINER,
            maintainer_email=MAINTAINER_EMAIL,
            description=DESCRIPTION,
            long_description=LONG_DESCRIPTION,
            url=URL,
            download_url=DOWNLOAD_URL,
            license=LICENSE,
            classifiers=CLASSIFIERS,
            author=AUTHOR,
            author_email=AUTHOR_EMAIL,
            platforms=PLATFORMS,
            version=VERSION,
            packages=PACKAGES,
            package_data={'bids': ['grabbids/config/*.json']},
            install_requires=REQUIRES,
            extras_require=EXTRAS_REQUIRE)


if __name__ == '__main__':
    setup(**opts)
