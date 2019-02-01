from .layout import BIDSLayout, add_config_paths
# Backwards compatibility
from bids_validator import BIDSValidator

__all__ = ["BIDSLayout", "BIDSValidator", "add_config_paths"]
