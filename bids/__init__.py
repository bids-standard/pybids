from .due import due, Doi
from .layout import BIDSLayout, BIDSLayoutIndexer

# For backwards compatibility
from bids_validator import BIDSValidator

__all__ = [
    "modeling",
    "BIDSLayout",
    "BIDSLayoutIndexer",
    "BIDSValidator",
    "config",
    "layout",
    "reports",
    "utils",
    "variables"
]

due.cite(Doi("10.1038/sdata.2016.44"),
         description="Brain Imaging Data Structure",
         tags=["reference-implementation"],
         path='bids')

del due, Doi

from . import _version
__version__ = _version.get_versions()['version']
