from __future__ import absolute_import, division, print_function
from .due import due, Doi

from .grabbids import BIDSLayout, BIDSValidator
from .analysis import Analysis
from .variables.io import load_variables


__all__ = [
    "grabbids",
    "analysis",
    "reports",
    "BIDSLayout",
    "BIDSValidator",
    "Analysis",
    "load_variables"
]

due.cite(Doi("10.1038/sdata.2016.44"),
         description="Brain Imaging Data Structure",
         tags=["reference-implementation"],
         path='bids')

del due, Doi

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
