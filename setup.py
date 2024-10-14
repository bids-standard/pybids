#!/usr/bin/env python
from setuptools import setup
import versioneer

setup(
    version="0.17.2-dev",
    cmdclass=versioneer.get_cmdclass(),
)
