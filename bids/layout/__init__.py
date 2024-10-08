import warnings

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
    "Query",
    "BIDSLayoutV2",
]

try:
    from .layout_v2 import BIDSLayoutV2
except Exception as err:
    def BIDSLayoutV2(*args, **kwargs):
        raise RuntimeError("Cannot create BIDSLayoutV2 - please install the ancpbids package.") from err
