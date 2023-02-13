import difflib
import enum
import os.path
import typing
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import List, Union, Dict, Optional, Any, Callable
import warnings

from .models import BIDSFile
from .utils import BIDSMetadata
from .writing import build_path, write_to_file
from ..external import inflect
from ..exceptions import (
    BIDSEntityError,
    BIDSValidationError,
    NoMatchError,
    TargetError,
)

from ancpbids import CustomOpExpr, EntityExpr, AllExpr, ValidationPlugin, load_dataset, validate_dataset, \
    write_derivative, DatasetOptions
from ancpbids.query import query, query_entities, FnMatchExpr, AnyExpr
from ancpbids.utils import deepupdate, resolve_segments, convert_to_relative, parse_bids_name

__all__ = ['BIDSLayout, Query']

from ..utils import natural_sort, listify


class BIDSLayoutWritingMixin:
    def build_path(self, source, path_patterns=None, strict=False,
                   scope='all', validate=True, absolute_paths=None):
        """Construct a target filename for a file or dictionary of entities.
        Parameters
        ----------
        source : str or :obj:`bids.layout.BIDSFile` or dict
            The source data to use to construct the new file path.
            Must be one of:
            - A BIDSFile object
            - A string giving the path of a BIDSFile contained within the
              current Layout.
            - A dict of entities, with entity names in keys and values in
              values
        path_patterns : list
            Optional path patterns to use to construct
            the new file path. If None, the Layout-defined patterns will
            be used. Entities should be represented by the name
            surrounded by curly braces. Optional portions of the patterns
            should be denoted by square brackets. Entities that require a
            specific value for the pattern to match can pass them inside
            angle brackets. Default values can be assigned by specifying a string
            after the pipe operator. E.g., (e.g., {type<image>|bold} would
            only match the pattern if the entity 'type' was passed and its
            value is "image", otherwise the default value "bold" will be
            used).
                Example: 'sub-{subject}/[var-{name}/]{id}.csv'
                Result: 'sub-01/var-SES/1045.csv'
        strict : bool, optional
            If True, all entities must be matched inside a
            pattern in order to be a valid match. If False, extra entities
            will be ignored so long as all mandatory entities are found.
        scope : str or list, optional
            The scope of the search space. Indicates which
            BIDSLayouts' path patterns to use. See BIDSLayout docstring
            for valid values. By default, uses all available layouts. If
            two or more values are provided, the order determines the
            precedence of path patterns (i.e., earlier layouts will have
            higher precedence).
        validate : bool, optional
            If True, built path must pass BIDS validator. If
            False, no validation is attempted, and an invalid path may be
            returned (e.g., if an entity value contains a hyphen).
        absolute_paths : bool, optional
            Optionally override the instance-wide option
            to report either absolute or relative (to the top of the
            dataset) paths. If None, will fall back on the value specified
            at BIDSLayout initialization.
        """
        raise NotImplementedError

    def copy_files(self, files=None, path_patterns=None, symbolic_links=True,
                   root=None, conflicts='fail', **kwargs):
        """Copy BIDSFile(s) to new locations.
        The new locations are defined by each BIDSFile's entities and the
        specified `path_patterns`.
        Parameters
        ----------
        files : list
            Optional list of BIDSFile objects to write out. If
            none provided, use files from running a get() query using
            remaining **kwargs.
        path_patterns : str or list
            Write patterns to pass to each file's write_file method.
        symbolic_links : bool
            Whether to copy each file as a symbolic link or a deep copy.
        root : str
            Optional root directory that all patterns are relative
            to. Defaults to dataset root.
        conflicts : str
            Defines the desired action when the output path already exists.
            Must be one of:
                'fail': raises an exception
                'skip' does nothing
                'overwrite': overwrites the existing file
                'append': adds a suffix to each file copy, starting with 1
        kwargs : dict
            Optional key word arguments to pass into a get() query.
        """
        raise NotImplementedError


    def write_to_file(self, entities, path_patterns=None,
                      contents=None, link_to=None, copy_from=None,
                      content_mode='text', conflicts='fail',
                      strict=False, validate=True):
        """Write data to a file defined by the passed entities and patterns.
        Parameters
        ----------
        entities : dict
            A dictionary of entities, with Entity names in
            keys and values for the desired file in values.
        path_patterns : list
            Optional path patterns to use when building
            the filename. If None, the Layout-defined patterns will be
            used.
        contents : object
            Contents to write to the generate file path.
            Can be any object serializable as text or binary data (as
            defined in the content_mode argument).
        link_to : str
            Optional path with which to create a symbolic link
            to. Used as an alternative to and takes priority over the
            contents argument.
        conflicts : str
            Defines the desired action when the output path already exists.
            Must be one of:
                'fail': raises an exception
                'skip' does nothing
                'overwrite': overwrites the existing file
                'append': adds a suffix to each file copy, starting with 1
        strict : bool
            If True, all entities must be matched inside a
            pattern in order to be a valid match. If False, extra entities
            will be ignored so long as all mandatory entities are found.
        validate : bool
            If True, built path must pass BIDS validator. If
            False, no validation is attempted, and an invalid path may be
            returned (e.g., if an entity value contains a hyphen).
        """
        raise NotImplementedError


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

class BIDSLayoutVariablesMixin:
    def get_collections(self, level, types=None, variables=None, merge=False,
                        sampling_rate=None, skip_empty=False, **kwargs):
        """Return one or more variable Collections in the BIDS project.
        Parameters
        ----------
        level : {'run', 'session', 'subject', 'dataset'}
            The level of analysis to return variables for.
            Must be one of 'run', 'session','subject', or 'dataset'.
        types : str or list
            Types of variables to retrieve. All valid values reflect the
            filename stipulated in the BIDS spec for each kind of variable.
            Valid values include: 'events', 'physio', 'stim', 'scans',
            'participants', 'sessions', and 'regressors'. Default is None.
        variables : list
            Optional list of variables names to return. If None, all available
            variables are returned.
        merge : bool
            If True, variables are merged across all observations of the
            current level. E.g., if level='subject', variables from all
            subjects will be merged into a single collection. If False, each
            observation is handled separately, and the result is returned
            as a list.
        sampling_rate : int or str
            If level='run', the sampling rate to pass onto the returned
            :obj:`bids.variables.collections.BIDSRunVariableCollection`.
        skip_empty : bool
            Whether or not to skip empty Variables (i.e., where there are no
            rows/records in a file after applying any filtering operations
            like dropping NaNs).
        kwargs
            Optional additional arguments to pass onto
            :obj:`bids.variables.io.load_variables`.
        Returns
        -------
        list of :obj:`bids.variables.collections.BIDSVariableCollection`
            or :obj:`bids.variables.collections.BIDSVariableCollection`
            A list if merge=False;
            a single :obj:`bids.variables.collections.BIDSVariableCollection`
            if merge=True.
        """

        from bids.variables import load_variables
        index = load_variables(self, types=types, levels=level,
                               skip_empty=skip_empty, **kwargs)
        return index.get_collections(level, variables, merge,
                                     sampling_rate=sampling_rate)



class BIDSLayout(BIDSLayoutMRIMixin, BIDSLayoutWritingMixin, BIDSLayoutVariablesMixin):
    """A convenience class to provide access to an in-memory representation of a BIDS dataset.

    .. code-block::

        dataset_path = 'path/to/your/dataset'
        layout = BIDSLayout(dataset_path)

    Parameters
    ----------
    ds_dir:
        the (absolute) path to the dataset to load
    """

    def __init__(
        self,
        root: Union[str, Path],
        validate: bool=True,
        derivatives: bool=False,
        config: Optional[Union[str, List[str]]]=None,
        sources: Optional[List[Any]]=None,
        config_filename: Optional[str]=None,
        regex_search: bool=False,
        database_path: Optional[str]=None,
        reset_database: Optional[bool]=None,
        indexer: Optional[Callable]=None,
        absolute_paths: Optional[bool]=None,
        ignore: Optional[List[str]]=None,
        force_index: Optional[List[str]]=None,
        **kwargs,
    ):
        if isinstance(root, Path):
            root = root.absolute()

        if ignore is None:
            if not (Path(root) / '.bidsignore').exists():
                ignore = ['.*', 'models', 'stimuli', 'code', 'sourcedata']
                warnings.warn(
                    """ No .bidsignore file found. Setting default ignore patterns.
                    In future versions of pybids a .bidsignore file will be 
                    required to ignore files. """,
                    DeprecationWarning
                )

        if force_index is not None:
            warnings.warn(
                "force_index no longer has any effect and will be removed",
                DeprecationWarning
            )

        options = DatasetOptions(ignore=ignore)

        self.dataset = load_dataset(root, options=options)
        self.schema = self.dataset.get_schema()
        self.validationReport = None

        self._regex_search = regex_search

        if validate:
            self.validationReport = self.validate()
            if self.validationReport.has_errors():
                error_message = os.linesep.join(map(lambda error: error['message'], self.validationReport.get_errors()))
                raise BIDSValidationError(error_message)

        if database_path is not None:
            warnings.warn(
                "database_path no longer has any effect and will be removed",
                DeprecationWarning
            )
        if reset_database is not None:
            warnings.warn(
                "reset_database no longer has any effect and will be removed",
                DeprecationWarning
            )
        if indexer is not None:
            warnings.warn(
                "indexer no longer has any effect and will be removed",
                DeprecationWarning
            )
        if absolute_paths is not None:
            warnings.warn(
                "absolute_paths no longer has any effect and will be removed",
                DeprecationWarning
            )
        if kwargs:
            warnings.warn(f"Unknown keyword arguments: {kwargs}")
        if config is not None:
            raise NotImplementedError("config is not implemented")
        if sources is not None:
            raise NotImplementedError("sources is not implemented")
        if config_filename is not None:
            raise NotImplementedError("config_filename is not implemented")

    def __getattr__(self, key):
        """Dynamically inspect missing methods for get_<entity>() calls
        and return a partial function of get() if a match is found."""
        try:
            return self.__dict__[key]
        except KeyError:
            pass
        if key.startswith('get_'):
            orig_ent_name = key.replace('get_', '')
            entities = self.get_entities(metadata=True)
            if ent_name not in entities:
                sing = inflect.engine().singular_noun(ent_name)
                if sing in entities:
                    ent_name = sing
                else:
                    raise BIDSEntityError(
                        "'get_{}' can't be called because '{}' isn't a "
                        "recognized entity name.".format(orig_ent_name, orig_ent_name))
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
        md = file.get_metadata(include_entities=include_entities)
        bmd = BIDSMetadata(file.get_absolute_path())
        bmd.update(md)
        return bmd

    def parse_file_entities(self, filename, scope=None, entities=None,
                            config=None, include_unmatched=False):
        """Parse the passed filename for entity/value pairs.
        Parameters
        ----------
        filename : str
            The filename to parse for entity values
        scope : str or list, optional
            The scope of the search space. Indicates which BIDSLayouts'
            entities to extract. See :obj:`bids.layout.BIDSLayout.get`
            docstring for valid values. By default, extracts all entities.
        entities : list or None, optional
            An optional list of Entity instances to use in
            extraction. If passed, the scope and config arguments are
            ignored, and only the Entities in this list are used.
        config : str or :obj:`bids.layout.models.Config` or list or None, optional
            One or more :obj:`bids.layout.models.Config` objects, or paths
            to JSON config files on disk, containing the Entity definitions
            to use in extraction. If passed, scope is ignored.
        include_unmatched : bool, optional
            If True, unmatched entities are included
            in the returned dict, with values set to None. If False
            (default), unmatched entities are ignored.
        Returns
        -------
        dict
            Dictionary where keys are Entity names and values are the
            values extracted from the filename.
        """
        results = parse_bids_name(filename)

        entities = results.pop('entities')
        schema_entities = {e.value['name']: e.name for e in list(self.schema.EntityEnum)}
        entities = {schema_entities[k]: v for k, v in entities.items()}
        results = {**entities, **results}

        if entities:
            # Filter out any entities that aren't in the specified
            results = {e: results[e] for e in entities if e in results}

        if include_unmatched:
            for k in set(self.get_entities()) - set(results):
                results[k] = None
        
        if scope is not None or config is not None:
            # To implement, need to be able to parse given a speciifc scope / config
            # Currently, parse_bids_name uses a fixed config in ancpbids
            raise NotImplementedError("scope and config are not implemented")

        return results

    def get_nearest(self, path, return_type='filename', strict=True,
                    all_=False, ignore_strict_entities='extension',
                    full_search=False, **filters):
        """Walk up file tree from specified path and return nearest matching file(s).
        Parameters
        ----------
        path (str): The file to search from.
        return_type (str): What to return; must be one of 'filename'
            (default) or 'tuple'.
        strict (bool): When True, all entities present in both the input
            path and the target file(s) must match perfectly. When False,
            files will be ordered by the number of matching entities, and
            partial matches will be allowed.
        all_ (bool): When True, returns all matching files. When False
            (default), only returns the first match.
        ignore_strict_entities (str, list): Optional entity/entities to
            exclude from strict matching when strict is True. This allows
            one to search, e.g., for files of a different type while
            matching all other entities perfectly by passing
            ignore_strict_entities=['type']. Ignores extension by default.
        full_search (bool): If True, searches all indexed files, even if
            they don't share a common root with the provided path. If
            False, only files that share a common root will be scanned.
        filters : dict
            Optional keywords to pass on to :obj:`bids.layout.BIDSLayout.get`.
        """
        
        path = Path(path).absolute()

        # Make sure we have a valid suffix
        if not filters.get('suffix'):
            f = self.get_file(path)
            if 'suffix' not in f.entities:
                raise BIDSValidationError(
                    "File '%s' does not have a valid suffix, most "
                    "likely because it is not a valid BIDS file." % path
                )
            filters['suffix'] = f.entities['suffix']

        # Collect matches for all entities for this file
        entities = self.parse_file_entities(str(path))

        # Remove any entities we want to ignore when strict matching is on
        if strict and ignore_strict_entities is not None:
            for k in listify(ignore_strict_entities):
                entities.pop(k, None)

        # Get candidate files
        results = self.get(**filters)

        # Make a dictionary of directories --> contained files
        folders = defaultdict(list)
        for f in results:
            folders[f._dirname].append(f)

        # Build list of candidate directories to check
        # Walking up from path, add all parent directories with a
        # matching file
        search_paths = []
        while True:
            if path in folders and folders[path]:
                search_paths.append(path)
            parent = path.parent
            if parent == path:
                break
            path = parent

        if full_search:
            unchecked = set(folders.keys()) - set(search_paths)
            search_paths.extend(path for path in unchecked if folders[path])

        def count_matches(f):
            # Count the number of entities shared with the passed file
            # Returns a tuple of (num_shared, num_perfect)
            f_ents = f.entities
            keys = set(entities.keys()) & set(f_ents.keys())
            shared = len(keys)
            return (shared, sum([entities[k] == f_ents[k] for k in keys]))

        matches = []

        for path in search_paths:
            # Sort by number of matching entities. Also store number of
            # common entities, for filtering when strict=True.
            num_ents = [(f, ) + count_matches(f) for f in folders[path]]
            # Filter out imperfect matches (i.e., where number of common
            # entities does not equal number of matching entities).
            if strict:
                num_ents = [f for f in num_ents if f[1] == f[2]]
            num_ents.sort(key=lambda x: x[2], reverse=True)

            if num_ents:
                matches += [f_match[0] for f_match in num_ents]

            if not all_:
                break

        matches = [match.path if return_type == 'filename'
                   else match for match in matches]
        return matches if all_ else matches[0] if matches else None
        

    def get(self, return_type: str = 'object', target: str = None, scope: str = None,
            extension: Union[str, List[str]] = None, suffix: Union[str, List[str]] = None,
            regex_search=None,
            **entities) -> Union[List[str], List[object]]:
        """Retrieve files and/or metadata from the current Layout.

        Parameters
        ----------
        return_type : str, optional
            Type of result to return. Valid values:
            'object' (default): return a list of matching BIDSFile objects.
            'file' or 'filename': return a list of matching filenames.
            'dir': return a list of directories.
            'id': return a list of unique IDs. Must be used together
                  with a valid target.
        target : str, optional
            Optional name of the target entity to get results for
            (only used if return_type is 'dir' or 'id').
        scope : str or list, optional
            Scope of the search space. If passed, only
            nodes/directories that match the specified scope will be
            searched. Possible values include:
            'all' (default): search all available directories.
            'derivatives': search all derivatives directories.
            'raw': search only BIDS-Raw directories.
            'self': search only the directly called BIDSLayout.
            <PipelineName>: the name of a BIDS-Derivatives pipeline.
        regex_search : bool or None, optional
            Whether to require exact matching
            (False) or regex search (True) when comparing the query string
            to each entity.
        invalid_filters (str): Controls behavior when named filters are
            encountered that don't exist in the database (e.g., in the case of
            a typo like subbject='0.1'). Valid values:
                'error' (default): Raise an explicit error.
                'drop': Silently drop invalid filters (equivalent to not having
                    passed them as arguments in the first place).
                'allow': Include the invalid filters in the query, resulting
                    in no results being returned.
        filters : dict
            Any optional key/values to filter the entities on.
            Keys are entity names, values are regexes to filter on. For
            example, passing filters={'subject': 'sub-[12]'} would return
            only files that match the first two subjects. In addition to
            ordinary data types, the following enums are defined (in the
            Query class):
                * Query.NONE: The named entity must not be defined.
                * Query.ANY: the named entity must be defined, but can have any
                    value.

        Returns
        -------
        list of :obj:`bids.layout.BIDSFile` or str
            A list of BIDSFiles (default) or strings (see return_type).
        """
        if regex_search is None:
            regex_search = self._regex_search

        # Provide some suggestions if target is specified and invalid.
        if return_type in ("dir", "id"):
            if target is None:
                raise TargetError(f'If return_type is "id" or "dir", a valid target '
                                  'entity must also be specified.')
            self_entities = self.get_entities()
            if target not in self_entities:
                potential = list(self_entities.keys())
                suggestions = difflib.get_close_matches(target, potential)
                if suggestions:
                    message = "Did you mean one of: {}?".format(suggestions)
                else:
                    message = "Valid targets are: {}".format(potential)
                raise TargetError(f"Unknown target '{target}'. {message}")  

        folder = self.dataset
        result = query(folder, return_type, target, scope, extension, suffix, regex_search, **entities)
        if return_type == 'file':
            result = natural_sort(result)
        if return_type == "object":
            if result:
                result = natural_sort(
                    [BIDSFile(res) for res in result],
                    "path"
                )
        return result

    @property
    def entities(self):
        return self.get_entities()

    def get_entities(self, scope: str = None, sort: bool = False, 
        long_form: bool = True, metadata: bool = True) -> dict:
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
        long_form: default is `True`
            whether to return the long form of the entity name (e.g., 'subject' instead of 'sub')

        Returns
        -------
        dict
            a unique set of entities found within the dataset as a dict
        """

        entities = query_entities(self.dataset, scope, long_form=long_form)

        if metadata is True:
            results = {**entities, **self._get_unique_metadata()}

        if sort:
            results = {k: sorted(v) for k, v in sorted(results.items())}

        return results

    def _get_unique_metadata(self):
        """Return a list of all unique metadata key and values found in the dataset."""
        
        all_metadata_objects = self.dataset.select(self.schema.MetadataArtifact).objects()

        metadata = defaultdict(set)
        for obj in all_metadata_objects:
            for k, v in obj['contents'].items():
                if isinstance(v, typing.Hashable):
                    metadata[k].add(v)

        return metadata

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

    def get_fieldmap(self, path, return_list=False):
        """Get fieldmap(s) for specified path."""
        fieldmaps = self._get_fieldmaps(path)

        if return_list:
            return fieldmaps
        else:
            if len(fieldmaps) == 1:
                return fieldmaps[0]
            elif len(fieldmaps) > 1:
                raise ValueError("More than one fieldmap found, but the "
                                 "'return_list' argument was set to False. "
                                 "Either ensure that there is only one "
                                 "fieldmap for this image, or set the "
                                 "'return_list' argument to True and handle "
                                 "the result as a list.")
            else:  # len(fieldmaps) == 0
                return None

    def _get_fieldmaps(self, path):
        path = str(path)
        sub = self.parse_file_entities(path)['subject']
        fieldmap_set = []
        suffix = '(phase1|phasediff|epi|fieldmap)'
        files = self.get(subject=sub, suffix=suffix, regex_search=True,
                         extension=['.nii.gz', '.nii'])
        for file in files:
            metadata = self.get_metadata(file.path)
            if metadata and "IntendedFor" in metadata.keys():
                intended_for = listify(metadata["IntendedFor"])
                # path uses local os separators while _suff read from json likely uses author's os separators, so we
                # convert _suff to use local separators.
                if any([path.endswith(str(Path(_suff))) for _suff in intended_for]):
                    cur_fieldmap = {}
                    if file.entities['suffix'] == "phasediff":
                        cur_fieldmap = {"phasediff": file.path,
                                        "magnitude1": file.path.replace(
                                            "phasediff", "magnitude1"),
                                        "suffix": "phasediff"}
                        magnitude2 = file.path.replace(
                            "phasediff", "magnitude2")
                        if Path(magnitude2).is_file():
                            cur_fieldmap['magnitude2'] = magnitude2
                    elif file.entities['suffix'] == "phase1":
                        cur_fieldmap["phase1"] = file.path
                        cur_fieldmap["magnitude1"] = \
                            file.path.replace("phase1", "magnitude1")
                        cur_fieldmap["phase2"] = \
                            file.path.replace("phase1", "phase2")
                        cur_fieldmap["magnitude2"] = \
                            file.path.replace("phase1", "magnitude2")
                        cur_fieldmap["suffix"] = "phase"
                    elif file.entities['suffix'] == "epi":
                        cur_fieldmap["epi"] = file.path
                        cur_fieldmap["suffix"] = "epi"
                    elif file.entities['suffix'] == "fieldmap":
                        cur_fieldmap["fieldmap"] = file.path
                        cur_fieldmap["magnitude"] = \
                            file.path.replace("fieldmap", "magnitude")
                        cur_fieldmap["suffix"] = "fieldmap"
                    fieldmap_set.append(cur_fieldmap)
        return fieldmap_set

    def add_derivatives(self, path):
        paths = listify(path)
        for path in paths:
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
        files = {file.path: file for file in all_files}
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
    def derivatives(self):
        derivatives = self.dataset.select(self.schema.DerivativeFolder).where(
            CustomOpExpr(lambda df: df.dataset_description is not None)).objects(as_list=True)
        # a dict where the key is the name of the derivative
        return {derivative.name: derivative for derivative in derivatives}

    @property
    def root(self):
        return self.dataset.base_dir_

    def __repr__(self):
        """Provide a tidy summary of key properties."""
        ents = self.get_entities()
        n_subjects = len(set(ents['subject'])) if 'subject' in ents else 0
        n_sessions = len(set(ents['session'])) if 'session' in ents else 0
        n_runs = len(set(ents['run'])) if 'run' in ents else 0
        s = ("BIDS Layout: ...{} | Subjects: {} | Sessions: {} | "
             "Runs: {}".format(self.dataset.base_dir_, n_subjects, n_sessions, n_runs))
        return s


class Query(enum.Enum):
    """Enums for use with BIDSLayout.get()."""
    NONE = 1 # Entity must not be present
    REQUIRED = ANY = 2  # Entity must be defined, but with an arbitrary value
    OPTIONAL = 3  # Entity may or may not be defined
