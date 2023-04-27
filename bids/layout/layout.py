"""BIDSLayout class."""
import re
from collections import defaultdict
from functools import partial, lru_cache
from itertools import chain
import json
import copy
import enum
import difflib
from pathlib import Path
import warnings
from typing import Hashable

import sqlalchemy as sa
from sqlalchemy.orm import aliased
from sqlalchemy.sql.expression import cast
from bids_validator import BIDSValidator

from ..utils import listify, natural_sort
from ..external import inflect
from ..exceptions import (
    BIDSDerivativesValidationError,
    BIDSEntityError,
    BIDSValidationError,
    NoMatchError,
    TargetError,
)

from .validation import (validate_root, validate_derivative_path,
                         absolute_path_deprecation_warning,
                         indexer_arg_deprecation_warning,
                         EXAMPLE_DERIVATIVES_DESCRIPTION)
from .writing import build_path, write_to_file
from .models import (Config, BIDSFile, DerivativeDatasets, Entity, Tag)
from .index import BIDSLayoutIndexer
from .db import ConnectionManager
from .utils import (BIDSMetadata, parse_file_entities)

__all__ = ['BIDSLayout']


class BIDSLayout:
    """Layout class representing an entire BIDS dataset.

    Parameters
    ----------
    root : str
        The root directory of the BIDS dataset.
    validate : bool, optional
        If True, all files are checked for BIDS compliance when first indexed,
        and non-compliant files are ignored. This provides a convenient way to
        restrict file indexing to only those files defined in the "core" BIDS
        spec, as setting validate=True will lead files in supplementary folders
        like derivatives/, code/, etc. to be ignored.
    absolute_paths : bool, optional
        If True, queries always return absolute paths.
        If False, queries return relative paths (for files and
        directories).
    derivatives : bool or str or list, optional
        Specifies whether and/or which
        derivatives to index. If True, all pipelines found in the
        derivatives/ subdirectory will be indexed. If a str or list, gives
        the paths to one or more derivatives directories to index. If False
        or None, the derivatives/ directory is ignored during indexing, and
        derivatives will have to be added manually via add_derivatives().
        Note: derivatives datasets MUST contain a dataset_description.json
        file in order to be indexed.
    config : str or list or None, optional
        Optional name(s) of configuration file(s) to use.
        By default (None), uses 'bids'.
    sources : :obj:`bids.layout.BIDSLayout` or list or None, optional
        Optional BIDSLayout(s) from which the current BIDSLayout is derived.
    config_filename : str
        Optional name of filename within directories
        that contains configuration information.
    regex_search : bool
        Whether to require exact matching (True) or regex
        search (False, default) when comparing the query string to each
        entity in .get() calls. This sets a default for the instance, but
        can be overridden in individual .get() requests.
    database_path : str
        Optional path to directory containing SQLite database file index
        for this BIDS dataset. If a value is passed and the folder
        already exists, indexing is skipped. By default (i.e., if None),
        an in-memory SQLite database is used, and the index will not
        persist unless .save() is explicitly called.
    reset_database : bool
        If True, any existing directory specified in the
        database_path argument is deleted, and the BIDS dataset provided
        in the root argument is reindexed. If False, indexing will be
        skipped and the existing database file will be used. Ignored if
        database_path is not provided.
    indexer: BIDSLayoutIndexer or callable
        An optional BIDSLayoutIndexer instance to use for indexing, or any
        callable that takes a BIDSLayout instance as its only argument. If
        None, a new indexer with default parameters will be implicitly created.
    is_derivative: bool
        Index dataset as a derivative dataset. This can be enabled along with
        validate=False to index derivatives without dataset_description.json. If
        validate=True, the dataset must have a dataset_description.json with
        DatasetType=derivative and GeneratedBy
    indexer_kwargs: dict
        Optional keyword arguments to pass onto the newly created
        BIDSLayoutIndexer. Valid keywords are 'ignore', 'force_index',
        'index_metadata', and 'config_filename'. Ignored if indexer is not
        None.
    """

    def __init__(self, root=None, validate=True, absolute_paths=True,
                 derivatives=False, config=None, sources=None,
                 regex_search=False, database_path=None, reset_database=False,
                 indexer=None, is_derivative=False, **indexer_kwargs):

        if not absolute_paths:
            absolute_path_deprecation_warning()

        ind_args = {'force_index', 'ignore', 'index_metadata', 'config_filename'}
        if ind_args & set(indexer_kwargs.keys()):
            indexer_arg_deprecation_warning()

        # Load from existing database file
        load_db = (database_path is not None and reset_database is False and
                   ConnectionManager.exists(database_path))

        if load_db:
            self.connection_manager = ConnectionManager(database_path)
            info = self.connection_manager.layout_info
            # Overwrite init args with values in DB
            root = info.root
            absolute_paths = info.absolute_paths
            derivatives = info.derivatives
            config = info.config

        # Validate that a valid BIDS project exists at root
        root, description = validate_root(root, validate)
        if any([
            is_derivative,
            description and description.get("DatasetType") == "derivative"
        ]):
            try:
                self.source_pipeline = validate_derivative_path(root)
            except BIDSValidationError as err:
                if validate:
                    raise err
                self.source_pipeline = None
            default_config = ["bids", "derivatives"]
            self.is_derivative = True
        else:
            self.source_pipeline = None
            default_config = ["bids"]
            self.is_derivative = False

        self._root = root  # type: Path
        self.description = description
        self.absolute_paths = absolute_paths
        self.derivatives = DerivativeDatasets()
        self.sources = sources
        self.regex_search = regex_search

        # Initialize a completely new layout and index the dataset
        if not load_db:
            # If config supplied from user or loaded from db, use it. Otherwise, fall
            # back to a default
            config = listify(config if config else default_config)

            init_args = dict(root=root, absolute_paths=absolute_paths,
                             derivatives=derivatives, config=config)

            self.connection_manager = ConnectionManager(
                database_path, reset_database, config, init_args)

            if indexer is None:
                indexer = BIDSLayoutIndexer(
                    validate=validate and not is_derivative, **indexer_kwargs
                )
            indexer(self)

        # Add derivatives if any are found
        if derivatives:
            if derivatives is True:
                derivatives = root / 'derivatives'
            self.add_derivatives(
                derivatives, parent_database_path=database_path,
                validate=validate, absolute_paths=absolute_paths,
                derivatives=None, sources=self, config=None,
                regex_search=regex_search, reset_database=reset_database,
                **indexer_kwargs)

    @property
    def root(self):
        return str(self._root)

    def __getattr__(self, key):
        """Dynamically inspect missing methods for get_<entity>() calls
        and return a partial function of get() if a match is found."""
        if key.startswith('get_'):
            ent_name = key.replace('get_', '')
            entities = self.get_entities()
            # Use inflect to check both singular and plural forms
            if ent_name not in entities:
                sing = inflect.engine().singular_noun(ent_name)
                if sing in entities:
                    ent_name = sing
                else:
                    raise BIDSEntityError(
                        "'get_{}' can't be called because '{}' isn't a "
                        "recognized entity name.".format(ent_name, ent_name))
            return partial(self.get, return_type='id', target=ent_name)
        # Spit out default message if we get this far
        raise AttributeError("%s object has no attribute named %r" %
                             (self.__class__.__name__, key))

    def __repr__(self):
        """Provide a tidy summary of key properties."""
        n_subjects = len(
            [s.value
             for s in self.session.query(Tag).filter_by(
                 entity_name='subject').group_by(Tag._value)]
            )

        n_sessions = len(
            set(
                (t.value, t.file.entities.get('subject'))
                 for t in
                 self.session.query(Tag).filter_by(entity_name='session')
                 if t.file.entities.get('subject')
                 )
            )

        n_runs = len(
            set(
                (t.value, t.file.entities.get('subject'))
                 for t in
                 self.session.query(Tag).filter_by(entity_name='run')
                 if isinstance(t.value, int) and t.file.entities.get('subject')
                 )
            )

        root = self.root[-30:]
        s = ("BIDS Layout: ...{} | Subjects: {} | Sessions: {} | "
             "Runs: {}".format(root, n_subjects, n_sessions, n_runs))
        return s

    def _in_scope(self, scope):
        """Determine whether current BIDSLayout is in the passed scope.

        Parameters
        ----------
        scope : str or list
            The intended scope(s). Each value must be one of 'all', 'raw',
            'derivatives', or a pipeline name.
        """
        scope = listify(scope)

        if 'all' in scope:
            return True

        # We assume something is a BIDS-derivatives dataset if it either has a
        # defined pipeline name, or is applying the 'derivatives' rules.
        pl_name = self.source_pipeline

        return any([
            'raw' in scope and not self.is_derivative,
            ('derivatives' in scope or pl_name in scope) and self.is_derivative
        ])

    def _get_layouts_in_scope(self, scope):
        """Return all layouts in the passed scope."""

        if scope == 'self':
            return [self]

        def collect_layouts(layout):
            """Recursively build a list of layouts."""
            children = list(layout.derivatives.values())
            layouts = [collect_layouts(d) for d in children]
            return [layout] + list(chain(*layouts))

        layouts = [l for l in collect_layouts(self) if l._in_scope(scope)]
        return list(set(layouts))

    def _sanitize_query_dtypes(self, entities):
        """Automatically convert entity query values to correct dtypes."""
        entities = entities.copy()
        names = list(entities.keys())
        ents = {e.name: e for e in
                self.session.query(Entity)
                    .filter(Entity.name.in_(names)).all()}
        # Fail silently because the DB may still know how to reconcile
        # type differences.
        for name, val in entities.items():
            if isinstance(val, enum.Enum):
                continue
            try:
                if isinstance(val, (list, tuple)):
                    entities[name] = [
                        v if isinstance(v, Query) else ents[name]._astype(v)
                        for v in val
                    ]
                else:
                    entities[name] = ents[name]._astype(val)
            except Exception:
                pass
        return entities

    @property
    def session(self):
        return self.connection_manager.session

    @property
    @lru_cache()
    def config(self):
        return {c.name: c for c in self.session.query(Config).all()}

    @property
    def entities(self):
        """Get the entities."""
        return self.get_entities()

    @property
    def files(self):
        """Get the files."""
        return self.get_files()

    @classmethod
    def load(cls, database_path):
        """ Load index from database path. Initialization parameters are set to
        those found in database_path JSON sidecar.

        Parameters
        ----------
        database_path : str, Path
            The path to the desired database folder. If a relative path is
            passed, it is assumed to be relative to the BIDSLayout root
            directory.
        """
        return cls(database_path=database_path)

    def save(self, database_path, replace_connection=True):
        """Save the current index as a SQLite3 DB at the specified location.

        Note: This is only necessary if a database_path was not specified
        at initialization, and the user now wants to save the index.
        If a database_path was specified originally, there is no need to
        re-save using this method.

        Parameters
        ----------
        database_path : str
            The path to the desired database folder. By default,
            uses .db_cache. If a relative path is passed, it is assumed to
            be relative to the BIDSLayout root directory.
        replace_connection : bool, optional
            If True, the newly created database will
            be used for all subsequent connections. This means that any
            changes to the index made after the .save() call will be
            reflected in the database file. If False, the previous database
            will continue to be used, and any subsequent changes will not
            be reflected in the new file unless save() is explicitly called
            again.
        """
        database_path = Path(database_path)

        self.connection_manager = self.connection_manager.save_database(
            database_path, replace_connection)

        # Recursively save children
        for pipeline_name, der in self.derivatives.items():
            der.save(database_path / pipeline_name)


    def get_entities(self, scope='all', metadata=None):
        """Get entities for all layouts in the specified scope.

        Parameters
        ----------
        scope : str
            The scope of the search space. Indicates which
            BIDSLayouts' entities to extract.
            See :obj:`bids.layout.BIDSLayout.get` docstring for valid values.

        metadata : bool or None
            By default (None), all available entities
            are returned. If True, only entities found in metadata files
            (and not defined for filenames) are returned. If False, only
            entities defined for filenames (and not those found in JSON
            sidecars) are returned.

        Returns
        -------
        dict
            Dictionary where keys are entity names and
            values are Entity instances.
        """
        # TODO: memoize results
        layouts = self._get_layouts_in_scope(scope)
        entities = {}
        for l in layouts:
            query = l.session.query(Entity)
            if metadata is not None:
                query = query.join(Tag).filter_by(is_metadata=metadata)
            results = query.all()
            entities.update({e.name: e for e in results})
        return entities

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
        # TODO: memoize results
        layouts = self._get_layouts_in_scope(scope)
        files = {}
        for l in layouts:
            results = l.session.query(BIDSFile).all()
            files.update({f.path: f for f in results})
        return files

    def clone(self):
        """Return a deep copy of the current BIDSLayout."""
        return copy.deepcopy(self)

    def parse_file_entities(self, filename, scope='all', entities=None,
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
        # If either entities or config is specified, just pass through
        if entities is None and config is None:
            layouts = self._get_layouts_in_scope(scope)
            config = chain(*[list(l.config.values()) for l in layouts])
            config = list(set(config))

        return parse_file_entities(filename, entities, config,
                                   include_unmatched)

    def _get_derivative_dirs(self, paths):
        for path in paths:
            p = Path(path).absolute()
            base = p.name
            if Path.exists(p / "dataset_description.json"):
                yield (base, p)
            else:
                yield from (
                    (f"{base}/{dd.parent.name}", dd.parent)
                    for dd in p.glob("*/dataset_description.json")
                )

    def add_derivatives(self, path, parent_database_path=None, **kwargs):
        """Add BIDS-Derivatives datasets to tracking.

        Parameters
        ----------
        path : str or list
            One or more paths to BIDS-Derivatives datasets.
            Each path can point to either a derivatives/ directory
            containing one or more pipeline directories, or to a single
            pipeline directory (e.g., derivatives/fmriprep).
        parent_database_path : str or Path
            If not None, use the pipeline name from the dataset_description.json
            file as the database folder name to nest within the parent database
            folder name to write out derivative index to.
        kwargs : dict
            Optional keyword arguments to pass on to
            BIDSLayout() when initializing each of the derivative datasets.

        Notes
        -----
        Every derivatives directory intended for indexing MUST contain a
        valid dataset_description.json file. See the BIDS-Derivatives
        specification for details.
        """
        paths = listify(path)
        if parent_database_path:
            parent_database_path = Path(parent_database_path)
        deriv_paths = list(self._get_derivative_dirs(paths))
        if not deriv_paths:
            warnings.warn("Derivative indexing was requested, but no valid "
                          "datasets were found in the specified locations "
                          "({}). Note that all BIDS-Derivatives datasets must"
                          " meet all the requirements for BIDS-Raw datasets "
                          "(a common problem is to fail to include a "
                          "'dataset_description.json' file in derivatives "
                          "datasets).\n".format(paths) +
                          "Example contents of 'dataset_description.json':\n%s" %
                          json.dumps(EXAMPLE_DERIVATIVES_DESCRIPTION))

        # Default config and sources values
        kwargs['sources'] = kwargs.get('sources') or self

        for name, path in deriv_paths:
            if parent_database_path:
                child_database_path = parent_database_path / name
                kwargs['database_path'] = child_database_path
            if name in self.derivatives:
                raise BIDSDerivativesValidationError(
                    f"Pipeline name {name} ({path!s}) has already been added to this "
                    "BIDSLayout. Every added pipeline must have a unique name!"
                )
            self.derivatives[name] = BIDSLayout(path, is_derivative=True, **kwargs)

    def to_df(self, metadata=False, **filters):
        """Return information for BIDSFiles tracked in Layout as pd.DataFrame.

        Parameters
        ----------
        metadata : bool, optional
            If True, includes columns for all metadata fields.
            If False, only filename-based entities are included as columns.
        filters : dict, optional
            Optional keyword arguments passed on to get(). This allows
            one to easily select only a subset of files for export.

        Returns
        -------
        :obj:`pandas.DataFrame`
            A pandas DataFrame, where each row is a file, and each column is
            a tracked entity. NaNs are injected whenever a file has no
            value for a given attribute.

        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError('Missing dependency: "pandas"')

        # TODO: efficiency could probably be improved further by joining the
        # BIDSFile and Tag tables and running a single query. But this would
        # require refactoring the below to use _build_file_query, which will
        # in turn likely require generalizing the latter.
        files = self.get(**filters)
        file_paths = [f.path for f in files]
        query = self.session.query(Tag).filter(Tag.file_path.in_(file_paths))

        if not metadata:
            query = query.join(Entity).filter(Tag.is_metadata == False)  # noqa: E712

        tags = query.all()

        tags = [[t.file_path, t.entity_name, t.value] for t in tags]
        data = pd.DataFrame(tags, columns=['path', 'entity', 'value'])
        data = data.pivot(index='path', columns='entity', values='value')

        # Add in orphaned files with no Tags. Maybe make this an argument?
        orphans = list(set(file_paths) - set(data.index))
        for o in orphans:
            data.loc[o] = pd.Series(dtype=float)

        return data.reset_index()
    def get(self, return_type='object', target=None, scope='all',
            regex_search=False, absolute_paths=None, invalid_filters='error',
            **filters):
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
        absolute_paths : bool, optional
            Optionally override the instance-wide option
            to report either absolute or relative (to the top of the
            dataset) paths. If None, will fall back on the value specified
            at BIDSLayout initialization.
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

        if absolute_paths is False:
            absolute_path_deprecation_warning()

        layouts = self._get_layouts_in_scope(scope)

        entities = self.get_entities()

        # error check on users accidentally passing in filters
        if isinstance(filters.get('filters'), dict):
            raise RuntimeError('You passed in filters as a dictionary named '
                               'filters; please pass the keys in as named '
                               'keywords to the `get()` call. For example: '
                               '`layout.get(**filters)`.')

        # Ensure leading periods if extensions were passed
        if 'extension' in filters and 'bids' in self.config:
            filters['extension'] = ['.' + x.lstrip('.') if isinstance(x, str) else x
                                    for x in listify(filters['extension'])]

        if invalid_filters != 'allow':
            bad_filters = set(filters.keys()) - set(entities.keys())
            if bad_filters:
                if invalid_filters == 'drop':
                    for bad_filt in bad_filters:
                        filters.pop(bad_filt)
                elif invalid_filters == 'error':
                    first_bad = list(bad_filters)[0]
                    msg = "'{}' is not a recognized entity. ".format(first_bad)
                    ents = list(entities.keys())
                    suggestions = difflib.get_close_matches(first_bad, ents)
                    if suggestions:
                        msg += "Did you mean {}? ".format(suggestions)
                    raise ValueError(msg + "If you're sure you want to impose "
                                     "this constraint, set "
                                     "invalid_filters='allow'.")

        # Provide some suggestions if target is specified and invalid.
        if target is not None and target not in entities:
            potential = list(entities.keys())
            suggestions = difflib.get_close_matches(target, potential)
            if suggestions:
                message = "Did you mean one of: {}?".format(suggestions)
            else:
                message = "Valid targets are: {}".format(potential)
            raise TargetError(("Unknown target '{}'. " + message)
                             .format(target))

        results = []
        for l in layouts:
            query = l._build_file_query(filters=filters,
                                        regex_search=regex_search)
            # NOTE: The following line, when uncommented, eager loads
            # associations. This was introduced in order to prevent sessions
            # from randomly detaching. It should be fixed by setting
            # expire_on_commit at session creation, but let's leave this here
            # for another release or two to make sure we don't have any further
            # problems.
            # query = query.options(joinedload(BIDSFile.tags)
            #                       .joinedload(Tag.entity))
            results.extend(query.all())

        # Convert to relative paths if needed
        if absolute_paths is None:  # can be overloaded as option to .get
            absolute_paths = self.absolute_paths

        if not absolute_paths:
            for i, fi in enumerate(results):
                fi = copy.copy(fi)
                fi.path = str(fi._path.relative_to(self._root))
                results[i] = fi

        if return_type.startswith('file'):
            results = natural_sort([f.path for f in results])

        elif return_type in ['id', 'dir']:
            if target is None:
                raise TargetError('If return_type is "id" or "dir", a valid '
                                 'target entity must also be specified.')

            if return_type == 'id':
                results = list(dict.fromkeys(
                    res.entities[target] for res in results
                    if target in res.entities and isinstance(res.entities[target], Hashable)
                ))

            elif return_type == 'dir':
                template = entities[target].directory
                if template is None:
                    raise ValueError('Return type set to directory, but no '
                                     'directory template is defined for the '
                                     'target entity (\"%s\").' % target)
                # Construct regex search pattern from target directory template
                # On Windows, the regex won't compile if, e.g., there is a folder starting with "U" on the path.
                # Converting to a POSIX path with forward slashes solves this.
                template = self._root.as_posix() + template
                to_rep = re.findall(r'{(.*?)\}', template)
                for ent in to_rep:
                    patt = entities[ent].pattern
                    template = template.replace('{%s}' % ent, patt)
                # Avoid matching subfolders. We are working with POSIX paths here, so we explicitly use "/"
                # as path separator.
                template += r'[^/]*$'
                matches = [
                    f.dirname if absolute_paths else str(f._dirname.relative_to(self._root))  # noqa: E501
                    for f in results
                    if re.search(template, f._dirname.as_posix())
                ]

                results = natural_sort(list(set(matches)))

            else:
                raise ValueError("Invalid return_type specified (must be one "
                                 "of 'tuple', 'filename', 'id', or 'dir'.")
        else:
            results = natural_sort(results, 'path')

        return results

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
        filename = self._root.joinpath(filename).absolute()
        for layout in self._get_layouts_in_scope(scope):
            result = layout.session.query(
                BIDSFile).filter_by(path=str(filename)).first()  # noqa: E501
            if result:
                return result
        return None

    def _build_file_query(self, **kwargs):

        query = self.session.query(BIDSFile).filter_by(is_dir=False)

        filters = kwargs.get('filters')

        # Entity filtering
        if filters:
            regex = kwargs.get('regex_search', False)

            filters = self._sanitize_query_dtypes(filters)

            for name, val in filters.items():
                tag_alias = aliased(Tag)

                val_list = list(listify(val)) if val is not None else [None]
                if not val_list:
                    continue

                if Query.OPTIONAL in val_list:
                    continue

                none = required = False
                if None in val_list:
                    none = True
                    val_list.remove(None)
                if Query.NONE in val_list:
                    none = True
                    val_list.remove(Query.NONE)
                if Query.REQUIRED in val_list:
                    required = True
                    val_list.remove(Query.REQUIRED)

                if none and required:
                    # Always true, apply no filter
                    continue

                # Baseline, use join, start accumulating clauses
                join_method = query.join
                val_clauses = []

                # NONE and REQUIRED get special treatment
                if none:
                    join_method = query.outerjoin
                    val_clauses.append(tag_alias._value.is_(None))
                if required:
                    val_clauses.append(tag_alias._value.isnot(None))

                # Any remaining values
                if regex:
                    val_clauses.extend([tag_alias._value.op('REGEXP')(str(v))
                                        for v in val_list])
                elif val_list:
                    _value = tag_alias._value
                    if isinstance(val_list[0], int):
                        _value = cast(tag_alias._value, sa.Integer)

                    val_clauses.append(_value.in_(val_list))

                # Looking for intersection with list of vals, so use OR
                val_clause = sa.or_(*val_clauses) if len(val_clauses) > 1 else val_clauses[0]

                query = join_method(
                    tag_alias,
                    sa.and_(
                        BIDSFile.path == tag_alias.file_path,
                        tag_alias.entity_name == name
                    ),
                )

                query = query.filter(val_clause)

        query = query.group_by(BIDSFile.path)

        return query

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
        md = BIDSMetadata(str(path))
        for layout in self._get_layouts_in_scope(scope):

            query = (layout.session.query(Tag)
                     .join(BIDSFile)
                     .filter(BIDSFile.path == str(path)))

            if not include_entities:
                query = query.join(Entity).filter(Tag.is_metadata == True)  # noqa: E712

            results = query.all()
            if results:
                md.update({t.entity_name: t.value for t in results})
                return md

        return md

    def get_dataset_description(self, scope='self', all_=False):
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
        layouts = self._get_layouts_in_scope(scope)
        if not all_:
            return layouts[0].get_file('dataset_description.json').get_dict()
        return [l.get_file('dataset_description.json').get_dict()
                for l in layouts]

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

        # Collect matches for all entities
        entities = {}
        for ent in self.get_entities(metadata=False).values():
            m = ent.regex.search(str(path))
            if m:
                entities[ent.name] = ent._astype(m.group(1))

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
            f_ents = f.entities
            keys = set(entities.keys()) & set(f_ents.keys())
            shared = len(keys)
            return [shared, sum([entities[k] == f_ents[k] for k in keys])]

        matches = []

        for path in search_paths:
            # Sort by number of matching entities. Also store number of
            # common entities, for filtering when strict=True.
            num_ents = [[f] + count_matches(f) for f in folders[path]]
            # Filter out imperfect matches (i.e., where number of common
            # entities does not equal number of matching entities).
            if strict:
                num_ents = [f for f in num_ents if f[1] == f[2]]
            num_ents.sort(key=lambda x: x[2], reverse=True)

            if num_ents:
                for f_match in num_ents:
                    matches.append(f_match[0])

            if not all_:
                break

        matches = [match.path if return_type.startswith('file')
                   else match for match in matches]
        return matches if all_ else matches[0] if matches else None

    def get_bvec(self, path, **kwargs):
        """Get bvec file for passed path."""
        result = self.get_nearest(path, extension='.bvec', suffix='dwi',
                                  all_=True, **kwargs)
        return listify(result)[0]

    def get_bval(self, path, **kwargs):
        """Get bval file for passed path."""
        result = self.get_nearest(path, suffix='dwi', extension='.bval',
                                  all_=True, **kwargs)
        return listify(result)[0]

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

    def get_tr(self, derivatives=False, **filters):

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
        filters.update(suffix='bold', datatype='func')
        scope = 'all' if derivatives else 'raw'
        images = self.get(extension=['.nii', '.nii.gz'], scope=scope,
                          **filters)
        if not images:
            raise NoMatchError("No functional images that match criteria found.")

        all_trs = set()
        for img in images:
            md = self.get_metadata(img.path)
            all_trs.add(round(float(md['RepetitionTime']), 5))

        if len(all_trs) > 1:
            raise NoMatchError("Unique TR cannot be found given filters {!r}"
                             .format(filters))
        return all_trs.pop()

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
        # 'is_file' is a crude check for Path objects
        if isinstance(source, str) or hasattr(source, 'is_file'):
            source = str(source)
            if source not in self.files:
                source = self._root / source

            source = self.get_file(source)

        if isinstance(source, BIDSFile):
            source = source.entities

        if path_patterns is None:
            layouts = self._get_layouts_in_scope(scope)
            path_patterns = []
            seen_configs = set()
            for l in layouts:
                for c in l.config.values():
                    if c in seen_configs:
                        continue
                    if c.default_path_patterns is not None:
                        path_patterns.extend(c.default_path_patterns)
                    seen_configs.add(c)

        built = build_path(source, path_patterns, strict)
        if built is None:
            raise ValueError(
                "Unable to construct build path with source {}".format(source))
        to_check = Path('/').joinpath(built).as_posix()

        if validate and not BIDSValidator().is_bids(to_check):
            raise BIDSValidationError(
                             "Built path {} is not a valid BIDS filename. "
                             "Please make sure all provided entity values are "
                             "spec-compliant.".format(built))

        # Convert to absolute paths if needed
        if absolute_paths is None:
            absolute_paths = self.absolute_paths

        if absolute_paths:
            built = self._root / built  # type: pathlib.Path
            # convert into a posix path for consistency with `writing.build_path`
            built = built.as_posix()  # type: str

        return built

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
        root = self.root if root is None else root

        _files = self.get(**kwargs)
        if files:
            _files = list(set(files).intersection(_files))

        for f in _files:
            f.copy(path_patterns, symbolic_link=symbolic_links,
                   root=root, conflicts=conflicts)

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
        path = self.build_path(entities, path_patterns, strict,
                               validate=validate)

        if path is None:
            raise ValueError("Cannot construct any valid filename for "
                             "the passed entities given available path "
                             "patterns.")

        write_to_file(path, contents=contents, link_to=link_to,
                      copy_from=copy_from, content_mode=content_mode,
                      conflicts=conflicts, root=self.root)


class Query(enum.Enum):
    """Enums for use with BIDSLayout.get()."""
    NONE = 1 # Entity must not be present
    REQUIRED = ANY = 2  # Entity must be defined, but with an arbitrary value
    OPTIONAL = 3  # Entity may or may not be defined
