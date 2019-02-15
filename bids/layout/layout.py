import os
import json
import re
from collections import defaultdict
from io import open
from functools import reduce, partial
from itertools import chain
import copy

from bids_validator import BIDSValidator
from ..utils import listify, natural_sort, check_path_matches_patterns
from ..external import inflect, six
from .core import Config, BIDSFile, BIDSRootNode
from .writing import build_path, write_contents_to_file
from .. import config as cf

try:
    from os.path import commonpath
except ImportError:
    def commonpath(paths):
        prefix = os.path.commonprefix(paths)
        if not os.path.isdir(prefix):
            prefix = os.path.dirname(prefix)
        return prefix


__all__ = ['BIDSLayout']


def parse_file_entities(filename, entities=None, config=None,
                        include_unmatched=False):
    """ Parse the passed filename for entity/value pairs.

    Args:
        filename (str): The filename to parse for entity values
        entities (list): An optional list of Entity instances to use in
            extraction. If passed, the config argument is ignored.
        config (str, Config, list): One or more Config objects or names of
            configurations to use in matching. Each element must be a Config
            object, or a valid Config name (e.g., 'bids' or 'derivatives').
            If None, all available configs are used.
        include_unmatched (bool): If True, unmatched entities are included
            in the returned dict, with values set to None. If False
            (default), unmatched entities are ignored.

    Returns: A dict, where keys are Entity names and values are the
        values extracted from the filename.
    """

    # Load Configs if needed
    if entities is None:

        if config is None:
            config = ['bids', 'derivatives']

        config = [Config.load(c) if not isinstance(c, Config) else c
                  for c in listify(config)]

        # Consolidate entities from all Configs into a single dict
        entities = {}
        for c in config:
            entities.update(c.entities)
        entities = entities.values()

    # Extract matches
    bf = BIDSFile(filename)
    ent_vals = {}
    for ent in entities:
        match = ent.match_file(bf)
        if match is not None or include_unmatched:
            ent_vals[ent.name] = match

    return ent_vals


def add_config_paths(**kwargs):
    """ Add to the pool of available configuration files for BIDSLayout.

    Args:
        kwargs: dictionary specifying where to find additional config files.
            Keys are names, values are paths to the corresponding .json file.

    Example:
        > add_config_paths(my_config='/path/to/config')
        > layout = BIDSLayout('/path/to/bids', config=['bids', 'my_config'])
    """

    for k, path in kwargs.items():
        if not os.path.exists(path):
            raise ValueError(
                'Configuration file "{}" does not exist'.format(k))
        if k in cf.get_option('config_paths'):
            raise ValueError('Configuration {!r} already exists'.format(k))

    kwargs.update(**cf.get_option('config_paths'))
    cf.set_option('config_paths', kwargs)


class BIDSLayout(object):
    """ Layout class representing an entire BIDS dataset.

    Args:
        root (str): The root directory of the BIDS dataset.
        validate (bool): If True, all files are checked for BIDS compliance
            when first indexed, and non-compliant files are ignored. This
            provides a convenient way to restrict file indexing to only those
            files defined in the "core" BIDS spec, as setting validate=True
            will lead files in supplementary folders like derivatives/, code/,
            etc. to be ignored.
        index_associated (bool): Argument passed onto the BIDSValidator;
            ignored if validate = False.
        absolute_paths (bool): If True, queries always return absolute paths.
            If False, queries return relative paths, unless the root argument
            was left empty (in which case the root defaults to the file system
            root).
        derivatives (bool, str, list): Specifies whether and/or which
            derivatives to to index. If True, all pipelines found in the
            derivatives/ subdirectory will be indexed. If a str or list, gives
            the paths to one or more derivatives directories to index. If False
            or None, the derivatives/ directory is ignored during indexing, and
            derivatives will have to be added manually via add_derivatives().
        config (str, list): Optional name(s) of configuration file(s) to use.
            By default (None), uses 'bids'.
        sources (BIDLayout, list): Optional BIDSLayout(s) from which the
            current BIDSLayout is derived.
        ignore (str, SRE_Pattern, list): Path(s) to exclude from indexing. Each
            path is either a string or a SRE_Pattern object (i.e., compiled
            regular expression). If a string is passed, it must be either an
            absolute path, or be relative to the BIDS project root. If an
            SRE_Pattern is passed, the contained regular expression will be
            matched against the full (absolute) path of all files and
            directories.
        force_index (str, SRE_Pattern, list): Path(s) to forcibly index in the
            BIDSLayout, even if they would otherwise fail validation. See the
            documentation for the ignore argument for input format details.
            Note that paths in force_index takes precedence over those in
            ignore (i.e., if a file matches both ignore and force_index, it
            *will* be indexed).
        config_filename (str): Optional name of filename within directories
            that contains configuration information.
        regex_search (bool): Whether to require exact matching (True) or regex
            search (False, default) when comparing the query string to each
            entity in .get() calls. This sets a default for the instance, but
            can be overridden in individual .get() requests.
    """

    _default_ignore = {"code", "stimuli", "sourcedata", "models",
                       "derivatives", re.compile(r'^\.')}

    def __init__(self, root, validate=True, index_associated=True,
                 absolute_paths=True, derivatives=False, config=None,
                 sources=None, ignore=None, force_index=None,
                 config_filename='layout_config.json', regex_search=False):

        self.root = root
        self._validator = BIDSValidator(index_associated=index_associated)
        self.validate = validate
        self.absolute_paths = absolute_paths
        self.derivatives = {}
        self.sources = sources
        self.regex_search = regex_search
        self.metadata_index = MetadataIndex(self)
        self.config_filename = config_filename
        self.files = {}
        self.nodes = []
        self.entities = {}
        self.ignore = [os.path.abspath(os.path.join(self.root, patt))
                       if isinstance(patt, six.string_types) else patt
                       for patt in listify(ignore or [])]
        self.force_index = [os.path.abspath(os.path.join(self.root, patt))
                            if isinstance(patt, six.string_types) else patt
                            for patt in listify(force_index or [])]

        # Do basic BIDS validation on root directory
        self._validate_root()

        # Initialize the BIDS validator and examine ignore/force_index args
        self._setup_file_validator()

        # Set up configs
        if config is None:
            config = 'bids'
        config = [Config.load(c) for c in listify(config)]
        self.config = {c.name: c for c in config}
        self.root_node = BIDSRootNode(self.root, config, self)

        # Consolidate entities into master list. Note: no conflicts occur b/c
        # multiple entries with the same name all point to the same instance.
        for n in self.nodes:
            self.entities.update(n.available_entities)

        # Add derivatives if any are found
        if derivatives:
            if derivatives is True:
                derivatives = os.path.join(root, 'derivatives')
            self.add_derivatives(
                derivatives, validate=validate,
                index_associated=index_associated,
                absolute_paths=absolute_paths, derivatives=None, config=None,
                sources=self, ignore=ignore, force_index=force_index)

    def _validate_root(self):
        # Validate root argument and make sure it contains mandatory info

        try:
            self.root = str(self.root)
        except:
            raise TypeError("root argument must be a string (or a type that "
                    "supports casting to string, such as pathlib.Path)"
                    " specifying the directory containing the BIDS dataset.")

        self.root = os.path.abspath(self.root)

        if not os.path.exists(self.root):
            raise ValueError("BIDS root does not exist: %s" % self.root)

        target = os.path.join(self.root, 'dataset_description.json')
        if not os.path.exists(target):
            if self.validate:
                raise ValueError(
                    "'dataset_description.json' is missing from project root."
                    " Every valid BIDS dataset must have this file.")
            else:
                self.description = None
        else:
            with open(target, 'r', encoding='utf-8') as desc_fd:
                self.description = json.load(desc_fd)
            if self.validate:
                for k in ['Name', 'BIDSVersion']:
                    if k not in self.description:
                        raise ValueError("Mandatory '%s' field missing from "
                                         "dataset_description.json." % k)

    def _setup_file_validator(self):
        # Derivatives get special handling; they shouldn't be indexed normally
        if self.force_index is not None:
            for entry in self.force_index:
                if (isinstance(entry, six.string_types) and
                    os.path.normpath(entry).startswith('derivatives')):
                        msg = ("Do not pass 'derivatives' in the force_index "
                               "list. To index derivatives, either set "
                               "derivatives=True, or use add_derivatives().")
                        raise ValueError(msg)

    def _validate_dir(self, d):
        return not check_path_matches_patterns(d, self.ignore)

    def _validate_file(self, f):
        # Validate a file.

        if check_path_matches_patterns(f, self.force_index):
            return True

        if check_path_matches_patterns(f, self.ignore):
            return False

        if not self.validate:
            return True

        # Derivatives are currently not validated.
        # TODO: raise warning the first time in a session this is encountered
        if 'derivatives' in self.config:
            return True

        # BIDS validator expects absolute paths, but really these are relative
        # to the BIDS project root.
        to_check = os.path.relpath(f, self.root)
        to_check = os.path.join(os.path.sep, to_check)

        return self._validator.is_bids(to_check)

    def _get_layouts_in_scope(self, scope):
        # Determine which BIDSLayouts to search
        layouts = []
        scope = listify(scope)
        if 'all' in scope or 'raw' in scope:
            layouts.append(self)
        for deriv in self.derivatives.values():
            if ('all' in scope or 'derivatives' in scope
                or deriv.description["PipelineDescription"]['Name'] in scope):
                layouts.append(deriv)
        return layouts
    
    def __getattr__(self, key):
        ''' Dynamically inspect missing methods for get_<entity>() calls
        and return a partial function of get() if a match is found. '''
        if key.startswith('get_'):
            ent_name = key.replace('get_', '')
            # Use inflect to check both singular and plural forms
            if ent_name not in self.entities:
                sing = inflect.engine().singular_noun(ent_name)
                if sing in self.entities:
                    ent_name = sing
                else:
                    raise AttributeError(
                        "'get_{}' can't be called because '{}' isn't a "
                        "recognized entity name.".format(ent_name, ent_name))
            return partial(self.get, return_type='id', target=ent_name)
        # Spit out default message if we get this far
        raise AttributeError("%s object has no attribute named %r" %
                             (self.__class__.__name__, key))

    def __repr__(self):
        # A tidy summary of key properties
        n_sessions = len([session for isub in self.get_subjects()
                          for session in self.get_sessions(subject=isub)])
        n_runs = len([run for isub in self.get_subjects()
                      for run in self.get_runs(subject=isub)])
        n_subjects = len(self.get_subjects())
        root = self.root[-30:]
        s = ("BIDS Layout: ...{} | Subjects: {} | Sessions: {} | "
             "Runs: {}".format(root, n_subjects, n_sessions, n_runs))
        return s

    def clone(self):
        """ Return a deep copy of the current BIDSLayout. """
        return copy.deepcopy(self)

    def parse_file_entities(self, filename, scope='all', entities=None,
                            config=None, include_unmatched=False):
        ''' Parse the passed filename for entity/value pairs.

        Args:
            filename (str): The filename to parse for entity values
            scope (str, list): The scope of the search space. Indicates which
                BIDSLayouts' entities to extract. See BIDSLayout docstring
                for valid values. By default, extracts all entities
            entities (list): An optional list of Entity instances to use in
                extraction. If passed, the scope and config arguments are
                ignored, and only the Entities in this list are used.
            config (str, Config, list): One or more Config objects, or paths
                to JSON config files on disk, containing the Entity definitions
                to use in extraction. If passed, scope is ignored.
            include_unmatched (bool): If True, unmatched entities are included
                in the returned dict, with values set to None. If False
                (default), unmatched entities are ignored.

        Returns: A dict, where keys are Entity names and values are the
            values extracted from the filename.
        '''

        # If either entities or config is specified, just pass through
        if entities is None and config is None:
            layouts = self._get_layouts_in_scope(scope)
            config = chain(*[list(l.config.values()) for l in layouts])
            config = list(set(config))

        return parse_file_entities(filename, entities, config,
                                   include_unmatched)

    def add_derivatives(self, path, **kwargs):
        ''' Add BIDS-Derivatives datasets to tracking.

        Args:
            path (str, list): One or more paths to BIDS-Derivatives datasets.
                Each path can point to either a derivatives/ directory
                containing one more more pipeline directories, or to a single
                pipeline directory (e.g., derivatives/fmriprep).
            kwargs (dict): Optional keyword arguments to pass on to
                BIDSLayout() when initializing each of the derivative datasets.
        '''
        paths = listify(path)
        deriv_dirs = []

        # Collect all paths that contain a dataset_description.json
        def check_for_description(dir):
            dd = os.path.join(dir, 'dataset_description.json')
            return os.path.exists(dd)

        for p in paths:
            p = os.path.abspath(p)
            if os.path.exists(p):
                if check_for_description(p):
                    deriv_dirs.append(p)
                else:
                    subdirs = [d for d in os.listdir(p)
                               if os.path.isdir(os.path.join(p, d))]
                    for sd in subdirs:
                        sd = os.path.join(p, sd)
                        if check_for_description(sd):
                            deriv_dirs.append(sd)

        for deriv in deriv_dirs:
            dd = os.path.join(deriv, 'dataset_description.json')
            with open(dd, 'r', encoding='utf-8') as ddfd:
                description = json.load(ddfd)
            pipeline_name = description.get(
                'PipelineDescription', {}).get('Name')
            if pipeline_name is None:
                raise ValueError("Every valid BIDS-derivatives dataset must "
                                 "have a PipelineDescription.Name field set "
                                 "inside dataset_description.json.")
            if pipeline_name in self.derivatives:
                raise ValueError("Pipeline name '%s' has already been added "
                                 "to this BIDSLayout. Every added pipeline "
                                 "must have a unique name!")
            # Default config and sources values
            kwargs['config'] = kwargs.get('config') or ['bids', 'derivatives']
            kwargs['sources'] = kwargs.get('sources') or self
            self.derivatives[pipeline_name] = BIDSLayout(deriv, **kwargs)

        # Consolidate all entities post-indexing. Note: no conflicts occur b/c
        # multiple entries with the same name all point to the same instance.
        for deriv in self.derivatives.values():
            self.entities.update(deriv.entities)

    def to_df(self, **kwargs):
        """
        Return information for all BIDSFiles tracked in the Layout as a pandas
        DataFrame.

        Args:
            kwargs: Optional keyword arguments passed on to get(). This allows
                one to easily select only a subset of files for export.
        Returns:
            A pandas DataFrame, where each row is a file, and each column is
                a tracked entity. NaNs are injected whenever a file has no
                value for a given attribute.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("What are you doing trying to export a BIDSLayout"
                              " as a pandas DataFrame when you don't have "
                              "pandas installed? Eh? Eh?")
        files = self.get(return_type='obj', **kwargs)
        data = pd.DataFrame.from_records([f.entities for f in files])
        data.insert(0, 'path', [f.path for f in files])
        return data

    def get(self, return_type='object', target=None, extensions=None,
            scope='all', regex_search=False, defined_fields=None, **kwargs):
        """
        Retrieve files and/or metadata from the current Layout.

        Args:
            return_type (str): Type of result to return. Valid values:
                'object' (default): return a list of matching BIDSFile objects.
                'file': return a list of matching filenames.
                'dir': return a list of directories.
                'id': return a list of unique IDs. Must be used together with
                    a valid target.
            target (str): Optional name of the target entity to get results for
                (only used if return_type is 'dir' or 'id').
            extensions (str, list): One or more file extensions to filter on.
                BIDSFiles with any other extensions will be excluded.
            scope (str, list): Scope of the search space. If passed, only
                nodes/directories that match the specified scope will be
                searched. Possible values include:
                    'all' (default): search all available directories.
                    'derivatives': search all derivatives directories
                    'raw': search only BIDS-Raw directories
                    <PipelineName>: the name of a BIDS-Derivatives pipeline
            regex_search (bool or None): Whether to require exact matching
                (False) or regex search (True) when comparing the query string
                to each entity.
            defined_fields (list): Optional list of names of metadata fields
                that must be defined in JSON sidecars in order to consider the
                file a match, but which don't need to match any particular
                value.
            kwargs (dict): Any optional key/values to filter the entities on.
                Keys are entity names, values are regexes to filter on. For
                example, passing filter={ 'subject': 'sub-[12]'} would return
                only files that match the first two subjects.

        Returns:
            A list of BIDSFiles (default) or strings (see return_type).
        """

        # Warn users still expecting 0.6 behavior
        if 'type' in kwargs:
            raise ValueError("As of pybids 0.7.0, the 'type' argument has been"
                             " replaced with 'suffix'.")

        layouts = self._get_layouts_in_scope(scope)
        
        # Create concatenated file, node, and entity lists
        files, entities, nodes = {}, {}, []
        for l in layouts:
            files.update(l.files)
            entities.update(l.entities)
            nodes.extend(l.nodes)

        # Separate entity kwargs from metadata kwargs
        ent_kwargs, md_kwargs = {}, {}
        for k, v in kwargs.items():
            if k in entities:
                ent_kwargs[k] = v
            else:
                md_kwargs[k] = v

        # Provide some suggestions if target is specified and invalid.
        if target is not None and target not in entities:
            import difflib
            potential = list(entities.keys())
            suggestions = difflib.get_close_matches(target, potential)
            if suggestions:
                message = "Did you mean one of: {}?".format(suggestions)
            else:
                message = "Valid targets are: {}".format(potential)
            raise ValueError(("Unknown target '{}'. " + message)
                             .format(target))

        results = []

        # Search on entities
        filters = ent_kwargs.copy()

        for f in files.values():
            if f._matches(filters, extensions, regex_search):
                results.append(f)

        # Search on metadata
        if return_type not in {'dir', 'id'}:

            if md_kwargs:
                results = [f.path for f in results]
                results = self.metadata_index.search(results, defined_fields,
                                                    **md_kwargs)
                results = [files[f] for f in results]

        # Convert to relative paths if needed
        if not self.absolute_paths:
            for i, f in enumerate(results):
                f = copy.copy(f)
                f.path = os.path.relpath(f.path, self.root)
                results[i] = f

        if return_type == 'file':
            results = natural_sort([f.path for f in results])

        elif return_type in ['id', 'dir']:
            if target is None:
                raise ValueError('If return_type is "id" or "dir", a valid '
                                 'target entity must also be specified.')
            results = [x for x in results if target in x.entities]

            if return_type == 'id':
                results = list(set([x.entities[target] for x in results]))
                results = natural_sort(results)

            elif return_type == 'dir':
                template = entities[target].directory
                if template is None:
                    raise ValueError('Return type set to directory, but no '
                                     'directory template is defined for the '
                                     'target entity (\"%s\").' % target)
                # Construct regex search pattern from target directory template
                template = template.replace('{{root}}', self.root)
                to_rep = re.findall(r'\{(.*?)\}', template)
                for ent in to_rep:
                    patt = entities[ent].pattern
                    template = template.replace('{%s}' % ent, patt)
                template += r'[^\%s]*$' % os.path.sep
                matches = [f.dirname for f in results
                           if re.search(template, f.dirname)]
                results = natural_sort(list(set(matches)))

            else:
                raise ValueError("Invalid return_type specified (must be one "
                                 "of 'tuple', 'file', 'id', or 'dir'.")

        return results

    def get_file(self, filename, scope='all'):
        ''' Returns the BIDSFile object with the specified path.

        Args:
            filename (str): The path of the file to retrieve. Must be either
                an absolute path, or relative to the root of this BIDSLayout.
            scope (str, list): Scope of the search space. If passed, only
                BIDSLayouts that match the specified scope will be
                searched. See BIDSLayout docstring for valid values.

        Returns: A BIDSFile, or None if no match was found.
        '''
        filename = os.path.abspath(os.path.join(self.root, filename))
        layouts = self._get_layouts_in_scope(scope)
        for ly in layouts:
            if filename in ly.files:
                return ly.files[filename]
        return None

    def get_collections(self, level, types=None, variables=None, merge=False,
                        sampling_rate=None, skip_empty=False, **kwargs):
        """Return one or more variable Collections in the BIDS project.

        Args:
            level (str): The level of analysis to return variables for. Must be
                one of 'run', 'session', 'subject', or 'dataset'.
            types (str, list): Types of variables to retrieve. All valid values
            reflect the filename stipulated in the BIDS spec for each kind of
            variable. Valid values include: 'events', 'physio', 'stim',
            'scans', 'participants', 'sessions', and 'regressors'.
            variables (list): Optional list of variables names to return. If
                None, all available variables are returned.
            merge (bool): If True, variables are merged across all observations
                of the current level. E.g., if level='subject', variables from
                all subjects will be merged into a single collection. If False,
                each observation is handled separately, and the result is
                returned as a list.
            sampling_rate (int, str): If level='run', the sampling rate to
                pass onto the returned BIDSRunVariableCollection.
            skip_empty (bool): Whether or not to skip empty Variables (i.e.,
                where there are no rows/records in a file after applying any
                filtering operations like dropping NaNs).
            kwargs: Optional additional arguments to pass onto load_variables.
        """
        from bids.variables import load_variables
        index = load_variables(self, types=types, levels=level,
                               skip_empty=skip_empty, **kwargs)
        return index.get_collections(level, variables, merge,
                                     sampling_rate=sampling_rate)

    def get_metadata(self, path, include_entities=False, **kwargs):
        """Return metadata found in JSON sidecars for the specified file.

        Args:
            path (str): Path to the file to get metadata for.
            include_entities (bool): If True, all available entities extracted
                from the filename (rather than JSON sidecars) are included in
                the returned metadata dictionary.
            kwargs (dict): Optional keyword arguments to pass onto
                get_nearest().

        Returns: A dictionary of key/value pairs extracted from all of the
            target file's associated JSON sidecars.

        Notes:
            A dictionary containing metadata extracted from all matching .json
            files is returned. In cases where the same key is found in multiple
            files, the values in files closer to the input filename will take
            precedence, per the inheritance rules in the BIDS specification.

        """

        f = self.get_file(path)

        # For querying efficiency, store metadata in the MetadataIndex cache
        self.metadata_index.index_file(f.path)

        if include_entities:
            entities = f.entities
            results = entities
        else:
            results = {}

        results.update(self.metadata_index.file_index[path])
        return results

    def get_nearest(self, path, return_type='file', strict=True, all_=False,
                    ignore_strict_entities=None, full_search=False, **kwargs):
        ''' Walk up the file tree from the specified path and return the
        nearest matching file(s).

        Args:
            path (str): The file to search from.
            return_type (str): What to return; must be one of 'file' (default)
                or 'tuple'.
            strict (bool): When True, all entities present in both the input
                path and the target file(s) must match perfectly. When False,
                files will be ordered by the number of matching entities, and
                partial matches will be allowed.
            all_ (bool): When True, returns all matching files. When False
                (default), only returns the first match.
            ignore_strict_entities (list): Optional list of entities to
                exclude from strict matching when strict is True. This allows
                one to search, e.g., for files of a different type while
                matching all other entities perfectly by passing
                ignore_strict_entities=['type'].
            full_search (bool): If True, searches all indexed files, even if
                they don't share a common root with the provided path. If
                False, only files that share a common root will be scanned.
            kwargs: Optional keywords to pass on to .get().
        '''

        path = os.path.abspath(path)

        # Make sure we have a valid suffix
        suffix = kwargs.get('suffix')
        if not suffix:
            f = self.get_file(path)
            if 'suffix' not in f.entities:
                raise ValueError(
                    "File '%s' does not have a valid suffix, most "
                    "likely because it is not a valid BIDS file." % path
                )
            suffix = f.entities['suffix']
        kwargs['suffix'] = suffix

        # Collect matches for all entities
        entities = {}
        for ent in self.entities.values():
            m = ent.regex.search(path)
            if m:
                entities[ent.name] = ent._astype(m.group(1))

        # Remove any entities we want to ignore when strict matching is on
        if strict and ignore_strict_entities is not None:
            for k in ignore_strict_entities:
                entities.pop(k, None)

        results = self.get(return_type='object', **kwargs)

        # Make a dictionary of directories --> contained files
        folders = defaultdict(list)
        for f in results:
            folders[f.dirname].append(f)

        # Build list of candidate directories to check
        search_paths = []
        while True:
            if path in folders and folders[path]:
                search_paths.append(path)
            parent = os.path.dirname(path)
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

        matches = [m.path if return_type == 'file' else m for m in matches]
        return matches if all_ else matches[0] if matches else None

    def get_bvec(self, path, **kwargs):
        """ Get bvec file for passed path. """
        result = self.get_nearest(path, extensions='bvec', suffix='dwi',
                                  all_=True, **kwargs)
        return listify(result)[0]

    def get_bval(self, path, **kwargs):
        """ Get bval file for passed path. """
        result = self.get_nearest(path, extensions='bval', suffix='dwi',
                                  all_=True, **kwargs)
        return listify(result)[0]

    def get_fieldmap(self, path, return_list=False):
        """ Get fieldmap(s) for specified path. """
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
        sub = self.parse_file_entities(path)['subject']
        fieldmap_set = []
        suffix = '(phase1|phasediff|epi|fieldmap)'
        files = self.get(subject=sub, suffix=suffix, regex_search=True,
                         extensions=['nii.gz', 'nii'])
        for file in files:
            metadata = self.get_metadata(file.path)
            if metadata and "IntendedFor" in metadata.keys():
                intended_for = listify(metadata["IntendedFor"])
                if any([path.endswith(_suff) for _suff in intended_for]):
                    cur_fieldmap = {}
                    if file.suffix == "phasediff":
                        cur_fieldmap = {"phasediff": file.path,
                                        "magnitude1": file.path.replace(
                                            "phasediff", "magnitude1"),
                                        "suffix": "phasediff"}
                        magnitude2 = file.path.replace(
                            "phasediff", "magnitude2")
                        if os.path.isfile(magnitude2):
                            cur_fieldmap['magnitude2'] = magnitude2
                    elif file.suffix == "phase1":
                        cur_fieldmap["phase1"] = file.path
                        cur_fieldmap["magnitude1"] = \
                            file.path.replace("phase1", "magnitude1")
                        cur_fieldmap["phase2"] = \
                            file.path.replace("phase1", "phase2")
                        cur_fieldmap["magnitude2"] = \
                            file.path.replace("phase1", "magnitude2")
                        cur_fieldmap["suffix"] = "phase"
                    elif file.suffix == "epi":
                        cur_fieldmap["epi"] = file.path
                        cur_fieldmap["suffix"] = "epi"
                    elif file.suffix == "fieldmap":
                        cur_fieldmap["fieldmap"] = file.path
                        cur_fieldmap["magnitude"] = \
                            file.path.replace("fieldmap", "magnitude")
                        cur_fieldmap["suffix"] = "fieldmap"
                    fieldmap_set.append(cur_fieldmap)
        return fieldmap_set

    def get_tr(self, derivatives=False, **selectors):
        """ Returns the scanning repetition time (TR) for one or more runs.

        Args:
            derivatives (bool): If True, also checks derivatives images.
            selectors: Optional keywords used to constrain the selected runs.
                Can be any arguments valid for a .get call (e.g., BIDS entities
                or JSON sidecar keys).
        
        Returns: A single float.

        Notes: Raises an exception if more than one unique TR is found.
        """
        # Constrain search to functional images
        selectors.update(suffix='bold', datatype='func')
        scope = None if derivatives else 'raw'
        images = self.get(extensions=['.nii', '.nii.gz'], scope=scope,
                          **selectors)
        if not images:
            raise ValueError("No functional images that match criteria found.")

        all_trs = set()
        for img in images:
            md = self.get_metadata(img.path, suffix='bold', full_search=True)
            all_trs.add(round(float(md['RepetitionTime']), 5))
 
        if len(all_trs) > 1:
            raise ValueError("Unique TR cannot be found given selectors {!r}"
                             .format(selectors))
        return all_trs.pop()

    def build_path(self, source, path_patterns=None, strict=False, scope='all'):
        ''' Constructs a target filename for a file or dictionary of entities.

        Args:
            source (str, BIDSFile, dict): The source data to use to construct
                the new file path. Must be one of:
                - A BIDSFile object
                - A string giving the path of a BIDSFile contained within the
                  current Layout.
                - A dict of entities, with entity names in keys and values in
                  values
            path_patterns (list): Optional path patterns to use to construct
                the new file path. If None, the Layout-defined patterns will
                be used.
            strict (bool): If True, all entities must be matched inside a
                pattern in order to be a valid match. If False, extra entities
                will be ignored so long as all mandatory entities are found.
            scope (str, list): The scope of the search space. Indicates which
                BIDSLayouts' path patterns to use. See BIDSLayout docstring
                for valid values. By default, uses all available layouts. If
                two or more values are provided, the order determines the
                precedence of path patterns (i.e., earlier layouts will have
                higher precedence).
        '''

        # 'is_file' is a crude check for Path objects
        if isinstance(source, six.string_types) or hasattr(source, 'is_file'):
            source = str(source)
            if source not in self.files:
                source = os.path.join(self.root, source)

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
                    path_patterns.extend(c.default_path_patterns)
                    seen_configs.add(c)

        return build_path(source, path_patterns, strict)

    def copy_files(self, files=None, path_patterns=None, symbolic_links=True,
                   root=None, conflicts='fail', **kwargs):
        """
        Copies one or more BIDSFiles to new locations defined by each
        BIDSFile's entities and the specified path_patterns.

        Args:
            files (list): Optional list of BIDSFile objects to write out. If
                none provided, use files from running a get() query using
                remaining **kwargs.
            path_patterns (str, list): Write patterns to pass to each file's
                write_file method.
            symbolic_links (bool): Whether to copy each file as a symbolic link
                or a deep copy.
            root (str): Optional root directory that all patterns are relative
                to. Defaults to current working directory.
            conflicts (str):  Defines the desired action when the output path
                already exists. Must be one of:
                    'fail': raises an exception
                    'skip' does nothing
                    'overwrite': overwrites the existing file
                    'append': adds  a suffix to each file copy, starting with 1
            kwargs (kwargs): Optional key word arguments to pass into a get()
                query.
        """
        _files = self.get(return_type='objects', **kwargs)
        if files:
            _files = list(set(files).intersection(_files))

        for f in _files:
            f.copy(path_patterns, symbolic_link=symbolic_links,
                   root=self.root, conflicts=conflicts)

    def write_contents_to_file(self, entities, path_patterns=None,
                               contents=None, link_to=None,
                               content_mode='text', conflicts='fail',
                               strict=False):
        """
        Write arbitrary data to a file defined by the passed entities and
        path patterns.

        Args:
            entities (dict): A dictionary of entities, with Entity names in
                keys and values for the desired file in values.
            path_patterns (list): Optional path patterns to use when building
                the filename. If None, the Layout-defined patterns will be
                used.
            contents (object): Contents to write to the generate file path.
                Can be any object serializable as text or binary data (as
                defined in the content_mode argument).
            link_to (str): Optional path with which to create a symbolic link
                to. Used as an alternative to and takes priority over the
                contents argument.
            conflicts (str):  Defines the desired action when the output path
                already exists. Must be one of:
                    'fail': raises an exception
                    'skip' does nothing
                    'overwrite': overwrites the existing file
                    'append': adds  a suffix to each file copy, starting with 1
            strict (bool): If True, all entities must be matched inside a
                pattern in order to be a valid match. If False, extra entities
        """
        path = self.build_path(entities, path_patterns, strict)

        if path is None:
            raise ValueError("Cannot construct any valid filename for "
                             "the passed entities given available path "
                             "patterns.")

        write_contents_to_file(path, contents=contents, link_to=link_to,
                               content_mode=content_mode, conflicts=conflicts,
                               root=self.root)


class MetadataIndex(object):
    """A simple dict-based index for key/value pairs in JSON metadata.

    Args:
        layout (BIDSLayout): The BIDSLayout instance to index.
    """

    def __init__(self, layout):
        self.layout = layout
        self.key_index = {}
        self.file_index = defaultdict(dict)

    def index_file(self, f, overwrite=False):
        """Index metadata for the specified file.

        Args:
            f (BIDSFile, str): A BIDSFile or path to an indexed file.
            overwrite (bool): If True, forces reindexing of the file even if
                an entry already exists.
        """
        if isinstance(f, six.string_types):
            f = self.layout.get_file(f)

        if f.path in self.file_index and not overwrite:
            return

        if 'suffix' not in f.entities:  # Skip files without suffixes
            return

        md = self._get_metadata(f.path)

        for md_key, md_val in md.items():
            if md_key not in self.key_index:
                self.key_index[md_key] = {}
            self.key_index[md_key][f.path] = md_val
            self.file_index[f.path][md_key] = md_val

    def _get_metadata(self, path, **kwargs):
        potential_jsons = listify(self.layout.get_nearest(
                                  path, extensions='.json', all_=True,
                                  ignore_strict_entities=['suffix'],
                                  **kwargs))

        if potential_jsons is None:
            return {}

        results = {}

        for json_file_path in reversed(potential_jsons):
            if os.path.exists(json_file_path):
                with open(json_file_path, 'r', encoding='utf-8') as fd:
                    param_dict = json.load(fd)
                results.update(param_dict)

        return results

    def search(self, files=None, defined_fields=None, **kwargs):
        """Search files in the layout by metadata fields.

        Args:
            files (list): Optional list of names of files to search. If None,
                all files in the layout are scanned.
            defined_fields (list): Optional list of names of fields that must
                be defined in the JSON sidecar in order to consider the file a
                match, but which don't need to match any particular value.
            kwargs: Optional keyword arguments defining search constraints;
                keys are names of metadata fields, and values are the values
                to match those fields against (e.g., SliceTiming=0.017) would
                return all files that have a SliceTiming value of 0.071 in
                metadata.

        Returns: A list of filenames that match all constraints.
        """

        if defined_fields is None:
            defined_fields = []

        all_keys = set(defined_fields) | set(kwargs.keys())
        if not all_keys:
            raise ValueError("At least one field to search on must be passed.")

        # If no list of files is passed, use all files in layout
        if files is None:
            files = set(self.layout.files.keys())

        # Index metadata for any previously unseen files
        for f in files:
            self.index_file(f)

        # Get file intersection of all kwargs keys--this is fast
        filesets = [set(self.key_index.get(k, [])) for k in all_keys]
        matches = reduce(lambda x, y: x & y, filesets)

        if files is not None:
            matches &= set(files)

        if not matches:
            return []

        def check_matches(f, key, val):
            if isinstance(val, six.string_types) and '*' in val:
                val = ('^%s$' % val).replace('*', ".*")
                return re.search(str(self.file_index[f][key]), val) is not None
            else:
                return val == self.file_index[f][key]

        # Serially check matches against each pattern, with early termination
        for k, val in kwargs.items():
            matches = list(filter(lambda x: check_matches(x, k, val), matches))
            if not matches:
                return []

        return matches
