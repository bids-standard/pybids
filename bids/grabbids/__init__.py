import warnings
from ..layout.bids_layout import BIDSLayout
from ..layout.bids_validator import BIDSValidator
__all__ = ["BIDSLayout", "BIDSValidator"]

warnings.warn("grabbids has been renamed to layout in version 0.6.5, and will be removed in version 0.8", FutureWarning)
