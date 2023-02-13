""" Model classes used in BIDSLayouts. """

import os
from typing import Union
from pathlib import Path
import json
from ancpbids.model_v1_8_0 import Artifact
from ancpbids.utils import parse_bids_name

from ..utils import listify
from .writing import build_path, write_to_file
from .utils import BIDSMetadata


class BIDSFile:
    """Represents a single file or directory in a BIDS dataset.

    Parameters
    ----------
    file_ref : str
        The path to the file or directory or an Artifact instance.
    """
    _ext_registry = {}

    def __init_subclass__(cls, extension, **kwargs):
        cls.extension = extension
        super().__init_subclass__(**kwargs)
        for ext in listify(extension):
            BIDSFile._ext_registry[ext] = cls

    @classmethod
    def from_filename(cls, filename):
        path = Path(filename)
        for ext, subclass in cls._ext_registry.items():
            if path.name.endswith(ext):
                cls = subclass
                break
        return cls(path)

    def __init__(self, file_ref: Union[str, os.PathLike, Artifact]):
        self._path = None
        self._artifact = None
        if isinstance(file_ref, (str, os.PathLike)):
            self._path = Path(file_ref)
        elif isinstance(file_ref, Artifact):
            self._artifact = file_ref
        
    @property
    def path(self):
        """ Convenience property for accessing path as a string."""
        try:
            return self._artifact.get_absolute_path()
        except AttributeError:
            return str(self._path)

    @property
    def filename(self):
        """ Convenience property for accessing filename."""
        try:
            return self._artifact.name
        except AttributeError:
            return self._path.name

    @property
    def is_dir(self):
        return Path(self.path).is_dir()

    @property
    def _dirname(self):
        return str(Path(self.path).parent)

    def __repr__(self):
        return f"<{self.__class__.__name__} filename='{self.path}'>"

    def __fspath__(self):
        return self.path

    @property
    def relpath(self):
        return self._artifact.get_relative_path()

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

    @property
    def entities(self):
        return self.get_entities()
        
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
        try:
            entities = self._artifact.get_entities()

            # Convert literal entity values to their names
            # schema_entities = {e.literal_: e.name for e in list(self.schema.EntityEnum)}
            # entities = {schema_entities[k]: v for k, v in entities.items()}
            entities['suffix'] = self._artifact.suffix
            entities['extension'] = self._artifact.extension

            if metadata:
                entities = {**entities, **self._artifact.get_metadata()}
        except AttributeError:
            raise NotImplementedError
        
        if values != 'tags':
            raise NotImplementedError

        return entities

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

        path = Path(self.path)
        
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
