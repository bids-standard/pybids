"""PyBIDS 1.0 API specification"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, TypeVar, Union

from .utils import PaddedInt

try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias


# Datasets should be parameterizable on some kind of schema object.
# External API users should not depend on it, so this is bound to Any,
# but once a Schema type is defined for an API implementation, type checkers
# should be able to introspect it.
SchemaT = TypeVar("SchemaT")


Index: TypeAlias = PaddedInt
Label: TypeAlias = str


class File(Protocol[SchemaT]):
    """Generic file holder

    This serves as a base class for :class:`BIDSFile` and can represent
    non-BIDS files.
    """

    path: Path
    dataset: Optional["BIDSDataset[SchemaT]"]

    def __fspath__(self) -> str:
        return str(self.path)

    @property
    def relative_path(self) -> Path:
        if self.dataset is None:
            raise ValueError("No dataset root to construct relative path from")
        return self.path.relative_to(self.dataset.root)


class BIDSFile(File[SchemaT], Protocol):
    """BIDS file

    This provides access to BIDS concepts such as path components
    and sidecar metadata.

    BIDS paths take the form::

        [sub-<label>/[ses-<label>/]<datatype>/]<entities>_<suffix><extension>
    """

    entities: Dict[str, Union[Label, Index]]
    datatype: Optional[str]
    suffix: Optional[str]
    extension: Optional[str]

    @property
    def metadata(self) -> Dict[str, Any]:
        """Sidecar metadata aggregated according to inheritance principle"""


class BIDSDataset(Protocol[SchemaT]):
    """Interface to a single BIDS dataset.

    This structure does not consider the contents of sub-datasets
    such as `sourcedata/` or `derivatives/`.
    """

    root: Path
    schema: SchemaT

    dataset_description: Dict[str, Any]
    """Contents of dataset_description.json"""

    ignored: List[File[SchemaT]]
    """Invalid files found in dataset"""

    files: List[BIDSFile[SchemaT]]
    """Valid files found in dataset"""

    datatypes: List[str]
    """Datatype directories found in dataset"""

    modalities: List[str]
    """BIDS "modalities" found in dataset"""

    subjects: List[str]
    """Subject/participant identifiers found in the dataset"""

    entities: List[str]
    """Entities (long names) found in any filename in the dataset"""

    def get(self, **filters) -> List[BIDSFile[SchemaT]]:
        """Query dataset for files"""

    def get_entities(self, entity: str, **filters) -> List[Label | Index]:
        """Query dataset for entity values"""

    def get_metadata(self, term: str, **filters) -> List[Any]:
        """Query dataset for metadata values"""


class DatasetCollection(BIDSDataset[SchemaT], Protocol):
    """Interface to a collection of BIDS dataset.

    This structure allows the user to construct a single view of
    multiple datasets, such as including source or derivative datasets.
    """

    primary: BIDSDataset[SchemaT]
    datasets: List[BIDSDataset[SchemaT]]

    def add_dataset(self, dataset: BIDSDataset[SchemaT]) -> None:
        ...
