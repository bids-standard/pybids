"""BIDSLayout class."""
import os
import re
from collections import defaultdict
from io import open
from functools import partial, lru_cache
from itertools import chain
import copy
import enum
import difflib
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy.orm import aliased
from sqlalchemy.sql.expression import cast
from bids_validator import BIDSValidator

from ..utils import listify, natural_sort
from ..external import inflect
from ..exceptions import (
    BIDSEntityError,
    BIDSValidationError,
    NoMatchError,
    TargetError,
)

from .validation import (validate_root, validate_derivative_paths,
                         absolute_path_deprecation_warning,
                         indexer_arg_deprecation_warning)
from .writing import build_path, write_to_file
from .models import (Config, BIDSFile, Entity, Tag)
from .index import BIDSLayoutIndexer
from .db import ConnectionManager
from .utils import (BIDSMetadata, parse_file_entities)

__all__ = ['BIDSLayout']


from ancpbids import BIDSLayout

class Query(enum.Enum):
    """Enums for use with BIDSLayout.get()."""
    NONE = 1 # Entity must not be present
    REQUIRED = ANY = 2  # Entity must be defined, but with an arbitrary value
    OPTIONAL = 3  # Entity may or may not be defined
