# Backwards compatibility
from bids_validator import BIDSValidator

from .index import BIDSLayoutIndexer
from .layout import BIDSLayout, Query
from .models import BIDSDataFile, BIDSFile, BIDSImageFile, BIDSJSONFile, Config, Entity, Tag
from .utils import add_config_paths, parse_file_entities

__all__ = [
    'BIDSLayout',
    'BIDSLayoutIndexer',
    'BIDSValidator',
    'add_config_paths',
    'parse_file_entities',
    'BIDSFile',
    'BIDSImageFile',
    'BIDSDataFile',
    'BIDSJSONFile',
    'Config',
    'Entity',
    'Tag',
    'Query',
]
