import warnings
from ..layout.bids_layout import BIDSLayout
from ..layout.bids_validator import BIDSValidator
__all__ = ["BIDSLayout", "BIDSValidator"]

warnings.simplefilter('always', DeprecationWarning)
warnings.warn("grabbids has been renamed to layout", DeprecationWarning)
