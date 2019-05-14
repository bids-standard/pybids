from .layout import BIDSLayout, add_config_paths, parse_file_entities
from .models import BIDSFile
# Backwards compatibility
from bids_validator import BIDSValidator

__all__ = ["BIDSLayout", "BIDSValidator", "add_config_paths",
           "parse_file_entities", "BIDSFile"]
