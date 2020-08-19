from .layout import BIDSLayout, Query
from .models import (BIDSFile, BIDSImageFile, BIDSDataFile, BIDSJSONFile,
                     Config, Entity, Tag)
from .index import BIDSLayoutIndexer
from .utils import add_config_paths, parse_file_entities
# Backwards compatibility
from bids_validator import BIDSValidator

__all__ = [
    "BIDSLayout",
    "BIDSLayoutIndexer",
    "BIDSValidator",
    "add_config_paths",
    "parse_file_entities",
    "BIDSFile",
    "BIDSImageFile",
    "BIDSDataFile",
    "BIDSJSONFile",
    "Config",
    "Entity",
    "Tag",
    "Query"
]
