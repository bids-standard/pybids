import difflib
import os.path
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import List, Union, Dict

from .utils import BIDSMetadata
from ..exceptions import (
    BIDSEntityError,
    BIDSValidationError,
    NoMatchError,
    TargetError,
)

from ancpbids import CustomOpExpr, EntityExpr, AllExpr, ValidationPlugin, load_dataset, validate_dataset, \
    write_derivative
from ancpbids.query import query, query_entities, FnMatchExpr, AnyExpr
from ancpbids.utils import deepupdate, resolve_segments, convert_to_relative

__all__ = ['BIDSLayoutV2']

class BIDSLayoutMRIMixin:
    def get_tr(self, derivatives=False, **entities):

        """Return the scanning repetition time (TR) for one or more runs.

        Parameters
        ----------
        derivatives : bool
            If True, also checks derivatives images.
        filters : dict
            Optional keywords used to constrain the selected runs.
            Can be any arguments valid for a .get call (e.g., BIDS entities
            or JSON sidecar keys).

        Returns
        -------
        float
            A single float.

        Notes
        -----
        Raises an exception if more than one unique TR is found.
        """
        # Constrain search to functional images
        scope = 'all' if derivatives else 'raw'
        images = self.get(scope=scope, extension=['.nii', '.nii.gz'], suffix='bold', **entities)
        if not images:
            raise NoMatchError("No functional images that match criteria found.")

        all_trs = set()
        for img in images:
            md = img.get_metadata()
            all_trs.add(round(float(md['RepetitionTime']), 5))

        if len(all_trs) > 1:
            raise NoMatchError("Unique TR cannot be found given filters {!r}"
                               .format(entities))
        return all_trs.pop()

class BIDSLayoutV2(BIDSLayoutMRIMixin):
    """A convenience class to provide access to an in-memory representation of a BIDS dataset.

    .. code-block::

        dataset_path = 'path/to/your/dataset'
        layout = BIDSLayout(dataset_path)

    Parameters
    ----------
    ds_dir:
        the (absolute) path to the dataset to load
    """

    def __init__(self, ds_dir: Union[str, Path], validate=True, **kwargs):
        if isinstance(ds_dir, Path):
            ds_dir = ds_dir.absolute()
        self.dataset = load_dataset(ds_dir)
        self.schema = self.dataset.get_schema()
        self.validationReport = None
        if validate:
            self.validationReport = self.validate()
            if self.validationReport.has_errors():
                error_message = os.linesep.join(map(lambda error: error['message'], self.validationReport.get_errors()))
                raise BIDSValidationError(error_message)

    def __getattr__(self, key):
        """Dynamically inspect missing methods for get_<entity>() calls
        and return a partial function of get() if a match is found."""
        if key.startswith('get_'):
            ent_name = key.replace('get_', '')
            ent_name = self.schema.fuzzy_match_entity_key(ent_name)
            return partial(self.get, return_type='id', target=ent_name)
        # Spit out default message if we get this far
        raise AttributeError("%s object has no attribute named %r" %
                             (self.__class__.__name__, key))

    def get_metadata(self, path, include_entities=False, scope='all'):
        """Return metadata found in JSON sidecars for the specified file.

        Parameters
        ----------
        path : str
            Path to the file to get metadata for.
        include_entities : bool, optional
            If True, all available entities extracted
            from the filename (rather than JSON sidecars) are included in
            the returned metadata dictionary.
        scope : str or list, optional
            The scope of the search space. Each element must
            be one of 'all', 'raw', 'self', 'derivatives', or a
            BIDS-Derivatives pipeline name. Defaults to searching all
            available datasets.

        Returns
        -------
        dict
            A dictionary of key/value pairs extracted from all of the
            target file's associated JSON sidecars.

        Notes
        -----
        A dictionary containing metadata extracted from all matching .json
        files is returned. In cases where the same key is found in multiple
        files, the values in files closer to the input filename will take
        precedence, per the inheritance rules in the BIDS specification.

        """
        path = convert_to_relative(self.dataset, path)
        file = self.dataset.get_file(path)
        md = file.get_metadata()
        if md and include_entities:
            schema_entities = {e.entity_: e.literal_ for e in list(self.schema.EntityEnum)}
            md.update({schema_entities[e.key]: e.value for e in file.entities})
        bmd = BIDSMetadata(file.get_absolute_path())
        bmd.update(md)
        return bmd

    def get(self, return_type: str = 'object', target: str = None, scope: str = None,
            extension: Union[str, List[str]] = None, suffix: Union[str, List[str]] = None,
            regex_search=False,
            **entities) -> Union[List[str], List[object]]:
        """Depending on the return_type value returns either paths to files that matched the filtering criteria
        or :class:`Artifact <ancpbids.model_v1_7_0.Artifact>` objects for further processing by the caller.

        Note that all provided filter criteria are AND combined, i.e. subj='02',task='lang' will match files containing
        '02' as a subject AND 'lang' as a task. If you provide a list of values for a criteria, they will be OR combined.

        .. code-block::

            file_paths = layout.get(subj='02', task='lang', suffix='bold', return_type='files')

            file_paths = layout.get(subj=['02', '03'], task='lang', return_type='files')

        Parameters
        ----------
        return_type:
            Either 'files' to return paths of matched files
            or 'object' to return :class:`Artifact <ancpbids.model_v1_7_0.Artifact>` object, defaults to 'object'

        target:
            Either `suffixes`, `extensions` or one of any valid BIDS entities key
            (see :class:`EntityEnum <ancpbids.model_v1_7_0.EntityEnum>`, defaults to `None`
        scope:
            a hint where to search for files
            If passed, only nodes/directories that match the specified scope will be
            searched. Possible values include:
            'all' (default): search all available directories.
            'derivatives': search all derivatives directories.
            'raw': search only BIDS-Raw directories.
            'self': search only the directly called BIDSLayout.
            <PipelineName>: the name of a BIDS-Derivatives pipeline.
        extension:
            criterion to match any files containing the provided extension only
        suffix:
            criterion to match any files containing the provided suffix only
        entities
            a list of key-values to match the entities of interest, example: subj='02',task='lang'

        Returns
        -------
            depending on the return_type value either paths to files that matched the filtering criteria
            or Artifact objects for further processing by the caller
        """
        # Provide some suggestions if target is specified and invalid.
        self_entities = self.get_entities()
        if target is not None and target not in self_entities:
            potential = list(self_entities.keys())
            suggestions = difflib.get_close_matches(target, potential)
            if suggestions:
                message = "Did you mean one of: {}?".format(suggestions)
            else:
                message = "Valid targets are: {}".format(potential)
            raise TargetError(("Unknown target '{}'. " + message)
                              .format(target))
        folder = self.dataset
        return query(folder, return_type, target, scope, extension, suffix, regex_search, **entities)

    @property
    def entities(self):
        return self.get_entities()

    def get_entities(self, scope: str = None, sort: bool = False) -> dict:
        """Returns a unique set of entities found within the dataset as a dict.
        Each key of the resulting dict contains a list of values (with at least one element).

        Example dict:
        .. code-block::

            {
                'sub': ['01', '02', '03'],
                'task': ['gamblestask']
            }

        Parameters
        ----------
        scope:
            see BIDSLayout.get()
        sort: default is `False`
            whether to sort the keys by name

        Returns
        -------
        dict
            a unique set of entities found within the dataset as a dict
        """
        return query_entities(self.dataset, scope, sort)

    def get_dataset_description(self, scope='self', all_=False) -> Union[List[Dict], Dict]:
        """Return contents of dataset_description.json.

        Parameters
        ----------
        scope : str
            The scope of the search space. Only descriptions of
            BIDSLayouts that match the specified scope will be returned.
            See :obj:`bids.layout.BIDSLayout.get` docstring for valid values.
            Defaults to 'self' --i.e., returns the dataset_description.json
            file for only the directly-called BIDSLayout.
        all_ : bool
            If True, returns a list containing descriptions for
            all matching layouts. If False (default), returns for only the
            first matching layout.

        Returns
        -------
        dict or list of dict
            a dictionary or list of dictionaries (depending on all_).
        """
        all_descriptions = self.dataset.select(self.schema.DatasetDescriptionFile).objects(as_list=True)
        if all_:
            return all_descriptions
        return all_descriptions[0] if all_descriptions else None

    def get_dataset(self) -> object:
        """
        Returns
        -------
            the in-memory representation of this layout/dataset
        """
        return self.dataset

    def add_derivatives(self, path):
        path = convert_to_relative(self.dataset, path)
        self.dataset.create_derivative(path=path)

    def write_derivative(self, derivative):
        """Writes the provided derivative folder to the dataset.
        Note that a 'derivatives' folder will be created if not present.

        Parameters
        ----------
        derivative:
            the derivative folder to write
        """
        assert isinstance(derivative, self.schema.DerivativeFolder)
        write_derivative(self.dataset, derivative)

    def validate(self) -> ValidationPlugin.ValidationReport:
        """Validates a dataset and returns a report object containing any detected validation errors.

        Example
        ----------

        .. code-block::

            report = layout.validate()
            for message in report.messages:
                print(message)
            if report.has_errors():
                raise "The dataset contains validation errors, cannot continue".

        Returns
        -------
            a report object containing any detected validation errors or warning
        """
        return validate_dataset(self.dataset)

    @property
    def files(self):
        return self.get_files()

    def get_files(self, scope='all'):
        """Get BIDSFiles for all layouts in the specified scope.

        Parameters
        ----------
        scope : str
            The scope of the search space. Indicates which
            BIDSLayouts' entities to extract.
            See :obj:`bids.layout.BIDSLayout.get` docstring for valid values.


        Returns:
            A dict, where keys are file paths and values
            are :obj:`bids.layout.BIDSFile` instances.

        """
        all_files = self.get(return_type="object", scope=scope)
        files = {file.get_absolute_path(): file for file in all_files}
        return files

    def get_file(self, filename, scope='all'):
        """Return the BIDSFile object with the specified path.

        Parameters
        ----------
        filename : str
            The path of the file to retrieve. Must be either an absolute path,
            or relative to the root of this BIDSLayout.
        scope : str or list, optional
            Scope of the search space. If passed, only BIDSLayouts that match
            the specified scope will be searched. See :obj:`BIDSLayout.get`
            docstring for valid values. Default is 'all'.

        Returns
        -------
        :obj:`bids.layout.BIDSFile` or None
            File found, or None if no match was found.
        """
        context = self.dataset
        filename = convert_to_relative(self.dataset, filename)
        if scope and scope not in ['all', 'raw', 'self']:
            context, _ = resolve_segments(context, scope)
        return context.get_file(filename)

    @property
    def description(self):
        return self.get_dataset_description()

    @property
    def root(self):
        return self.dataset.base_dir_

    def __repr__(self):
        """Provide a tidy summary of key properties."""
        ents = self.get_entities()
        n_subjects = len(set(ents['sub'])) if 'sub' in ents else 0
        n_sessions = len(set(ents['ses'])) if 'ses' in ents else 0
        n_runs = len(set(ents['run'])) if 'run' in ents else 0
        s = ("BIDS Layout: ...{} | Subjects: {} | Sessions: {} | "
             "Runs: {}".format(self.dataset.base_dir_, n_subjects, n_sessions, n_runs))
        return s



