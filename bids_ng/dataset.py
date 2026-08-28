import os
import re
import typing as ty
from functools import cached_property
from pathlib import Path

import bidsschematools as bst  # type: ignore[import]
import bidsschematools.schema  # type: ignore[import]
import bidsschematools.types  # type: ignore[import]

from . import types as bt


class BIDSValidationError(ValueError):
    """Error arising from invalid files or values in a BIDS dataset"""


class Schema:
    schema: bst.types.Namespace

    def __init__(
        self,
        schema: ty.Union[bst.types.Namespace, None] = None,
    ):
        if schema is None:
            # Bundled
            schema = bst.schema.load_schema()
        self.schema = schema

    @classmethod
    def from_spec(cls, schema_spec: str) -> "Schema":
        return cls(bst.schema.load_schema(schema_spec))

    # Conveniences to avoid `schema.schema` pattern
    @property
    def objects(self) -> bst.types.Namespace:
        return self.schema.objects

    @property
    def rules(self) -> bst.types.Namespace:
        return self.schema.rules

    @property
    def meta(self) -> bst.types.Namespace:
        return self.schema.meta


default_schema = Schema()


class File(bt.File[Schema]):
    """Generic file holder

    This serves as a base class for :class:`BIDSFile` and can represent
    non-BIDS files.
    """

    def __init__(
        self,
        path: ty.Union[os.PathLike, str],
        dataset: ty.Optional["BIDSDataset"] = None,
    ):
        self.path = Path(path)
        self.dataset = dataset


class BIDSFile(File, bt.BIDSFile[Schema]):
    """BIDS file"""

    pattern = re.compile(
        r"""
        (?:(?P<entities>(?:[a-z]+-[a-zA-Z0-9]+(?:_[a-z]+-[a-zA-Z0-9]+)*))_)?
        (?P<suffix>[a-zA-Z0-9]+)
        (?P<extension>\.[^/\\]+)$
        """,
        re.VERBOSE,
    )

    def __init__(
        self,
        path: ty.Union[os.PathLike, str],
        dataset: ty.Optional["BIDSDataset"] = None,
    ):
        super().__init__(path, dataset)
        self.entities = {}
        self.datatype = None
        self.suffix = None
        self.extension = None

        schema = default_schema if dataset is None else dataset.schema

        if self.path.parent.name in schema.objects.datatypes:
            self.datatype = self.path.parent.name

        matches = self.pattern.match(self.path.name)
        if matches is None:
            return

        entities, self.suffix, self.extension = matches.groups()

        if entities:
            found_entities = dict(ent.split("-") for ent in entities.split("_"))
            self.entities = {
                key: bt.Index(value) if entity.format == "index" else value
                for key, entity in schema.rules.entities.items()
                if (value := found_entities.get(entity.name)) is not None
            }

    @cached_property
    def metadata(self) -> dict[str, ty.Any]:
        """Sidecar metadata aggregated according to inheritance principle"""
        if not self.dataset:
            raise ValueError
        # TODO
        return {}


class BIDSDataset(bt.BIDSDataset[Schema]):
    ...
