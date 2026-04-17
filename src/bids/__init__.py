# For backwards compatibility  # noqa: D104
from bids_validator import BIDSValidator

from .due import Doi, due
from .layout import BIDSLayout, BIDSLayoutIndexer

__all__ = [
    'modeling',
    'BIDSLayout',
    'BIDSLayoutIndexer',
    'BIDSValidator',
    'config',
    'layout',
    'reports',
    'utils',
    'variables',
]

due.cite(
    Doi('10.1038/sdata.2016.44'),
    description='Brain Imaging Data Structure',
    tags=['reference-implementation'],
    path='bids',
)

del due, Doi

from . import _version  # noqa: E402

__version__ = _version.get_versions()['version']
