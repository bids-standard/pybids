""" Model classes used in BIDSLayouts. """

import re
import os
from pathlib import Path
import warnings
import json
from copy import deepcopy
from itertools import chain
from functools import lru_cache

from ..utils import listify
from .writing import build_path, write_to_file
from ..config import get_option
from .utils import BIDSMetadata, PaddedInt


class BIDSFile:
    """Represents a single file or directory in a BIDS dataset.

    Parameters
    ----------
    filename : str
        The path to the corresponding file.
    """
    _ext_registry = {}

    def __init_subclass__(cls, extension, **kwargs):
        cls.extension = extension
        super().__init_subclass__(**kwargs)
        for ext in listify(extension):
            BIDSFile._ext_registry[ext] = cls

    @classmethod
    def from_path(cls, path):
        path = Path(path)
        for ext, subclass in cls._ext_registry.items():
            if path.name.endswith(ext):
                cls = subclass
                break
        return cls(path)

    def __init__(self, filename, root=None):
        self.path = str(filename)
        self.filename = self._path.name
        self.dirname = str(self._path.parent)
        self.is_dir = not self.filename
        self._root = root

    @property
    def _path(self):
        return Path(self.path)

    def __repr__(self):
        return f"<{self.__class__.__name__} filename='{self.path}'>"

    def __fspath__(self):
        return self.path

    @property
    def relpath(self):
        """Return path relative to layout root"""
        return str(Path(self.path).relative_to(self._root))

    def get_associations(self, kind=None, include_parents=False):
        """Get associated files, optionally limiting by association kind.

        Parameters
        ----------
        kind : str
            The kind of association to return (e.g., "Child").
            By default, all associations are returned.
        include_parents : bool
            If True, files related through inheritance
            are included in the returned list. If False, only directly
            associated files are returned. For example, a file's JSON
            sidecar will always be returned, but other JSON files from
            which the sidecar inherits will only be returned if
            include_parents=True.

        Returns
        -------
        list
            A list of BIDSFile instances.
        """
        raise NotImplementedError

    def get_metadata(self):
        """Return all metadata associated with the current file. """
        md = BIDSMetadata(self.path)
        md.update(self.get_entities(metadata=True))
        return md

    def get_entities(self, metadata=False, values='tags'):
        """Return entity information for the current file.

        Parameters
        ----------
        metadata : bool or None
            If False (default), only entities defined
            for filenames (and not those found in the JSON sidecar) are
            returned. If True, only entities found in metadata files (and not
            defined for filenames) are returned. If None, all available
            entities are returned.
        values : str
            The kind of object to return in the dict's values.
            Must be one of:
                * 'tags': Returns only the tagged value--e.g., if the key
                is "subject", the value might be "01".
                * 'objects': Returns the corresponding Entity instance.

        Returns
        -------
        dict
            A dict, where keys are entity names and values are Entity
            instances.
        """
        session = object_session(self)
        query = (session.query(Tag)
                 .filter_by(file_path=self.path)
                 .join(Entity))

        if metadata not in (None, 'all'):
            query = query.filter(Tag.is_metadata == metadata)

        results = query.all()
        if values.startswith('obj'):
            return {t.entity_name: t.entity for t in results}
        return {t.entity_name: t.value for t in results}

    def copy(self, path_patterns, symbolic_link=False, root=None,
             conflicts='fail'):
        """Copy the contents of a file to a new location.

        Parameters
        ----------
        path_patterns : list
            List of patterns used to construct the new
            filename. See :obj:`build_path` documentation for details.
        symbolic_link : bool
            If True, use a symbolic link to point to the
            existing file. If False, creates a new file.
        root : str
            Optional path to prepend to the constructed filename.
        conflicts : str
            Defines the desired action when the output path already exists.
            Must be one of:
                'fail': raises an exception
                'skip' does nothing
                'overwrite': overwrites the existing file
                'append': adds  a suffix to each file copy, starting with 1
        """
        new_filename = build_path(self.entities, path_patterns)
        if not new_filename:
            return None

        if new_filename[-1] == os.sep:
            new_filename += self.filename

        if self._path.is_absolute() or root is None:
            path = self._path
        else:
            path = Path(root) / self._path

        if not path.exists():
            raise ValueError("Target filename to copy/symlink (%s) doesn't "
                             "exist." % path)

        kwargs = dict(path=new_filename, root=root, conflicts=conflicts)
        if symbolic_link:
            kwargs['link_to'] = path
        else:
            kwargs['copy_from'] = path

        write_to_file(**kwargs)


class BIDSDataFile(BIDSFile, extension=[".tsv", ".tsv.gz"]):
    """Represents a single data file in a BIDS dataset.

    Derived from `BIDSFile` and provides additional functionality such as
    obtaining pandas DataFrame data representation (via `get_df`).
    """

    def get_df(self, include_timing=True, adjust_onset=False,
               enforce_dtypes=True, **pd_args):
        """Return the contents of a tsv file as a pandas DataFrame.

        Parameters
        ----------
        include_timing : bool
            If True, adds an "onset" column to dense
            timeseries files (e.g., *_physio.tsv.gz).
        adjust_onset : bool
            If True, the onset of each sample in a dense
            timeseries file is shifted to reflect the "StartTime" value in
            the JSON sidecar. If False, the first sample starts at 0 secs.
            Ignored if include_timing=False.
        enforce_dtypes : bool
            If True, enforces the data types defined in
            the BIDS spec on core columns (e.g., subject_id and session_id
            must be represented as strings).
        pd_args : dict
            Optional keyword arguments to pass onto pd.read_csv().

        Returns
        -------
        :obj:`pandas.DataFrame`
            A pandas DataFrame.
        """
        import pandas as pd
        import numpy as np

        if enforce_dtypes:
            dtype = {
                'subject_id': str,
                'session_id': str,
                'participant_id': str
            }
        else:
            dtype = None

        # TODO: memoize this for efficiency. (Note: caching is insufficient,
        # because the dtype enforcement will break if we ignore the value of
        # enforce_dtypes).
        suffix = self.entities['suffix']
        header = None if suffix in {'physio', 'stim'} else 'infer'
        self.data = pd.read_csv(self.path, sep='\t', na_values='n/a',
                                dtype=dtype, header=header, **pd_args)

        data = self.data.copy()

        if self.entities['extension'] == '.tsv.gz':
            md = self.get_metadata()
            # We could potentially include some validation here, but that seems
            # like a job for the BIDS Validator.
            data.columns = md['Columns']
            if include_timing:
                onsets = np.arange(len(data)) / md['SamplingFrequency']
                if adjust_onset:
                    onsets += md['StartTime']
                data.insert(0, 'onset', onsets)

        return data


class BIDSImageFile(BIDSFile, extension=[".nii", ".nii.gz", ".gii"]):
    """Represents a single neuroimaging data file in a BIDS dataset.

    Derived from `BIDSFile` and provides additional functionality such as
    obtaining nibabel's image file representation (via `get_image`).
    """

    def get_image(self, **kwargs):
        """Return the associated image file (if it exists) as a NiBabel object

        Any keyword arguments are passed to ``nibabel.load``.
        """
        try:
            import nibabel as nb
            return nb.load(self.path, **kwargs)
        except Exception as e:
            raise ValueError("'{}' does not appear to be an image format "
                             "NiBabel can read.".format(self.path)) from e


class BIDSJSONFile(BIDSFile, extension=".json"):
    """Represents a single JSON metadata file in a BIDS dataset.

    Derived from `BIDSFile` and provides additional functionality for reading
    the contents of JSON files as either dicts or strings.
    """
    def get_dict(self):
        """Return the contents of the current file as a dictionary. """
        d = json.loads(self.get_json())
        if not isinstance(d, dict):
            raise ValueError("File %s is a json containing %s, not a dict which was expected" % (self.path, type(d)))
        return d

    def get_json(self):
        """Return the contents of the current file as a JSON string. """
        with open(self.path, 'r') as f:
            return f.read()
