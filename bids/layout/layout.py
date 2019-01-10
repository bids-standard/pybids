import os
import re
import json
import warnings
from io import open
from .validation import BIDSValidator
from .. import config as cf
from grabbit import Layout, File
from grabbit.external import six, inflect
from grabbit.utils import listify
from collections import defaultdict
from functools import reduce, partial
from itertools import chain
from bids.config import get_option


try:
    from os.path import commonpath
except ImportError:
    def commonpath(paths):
        prefix = os.path.commonprefix(paths)
        if not os.path.isdir(prefix):
            prefix = os.path.dirname(prefix)
        return prefix


__all__ = ['BIDSLayout']


def add_config_paths(**kwargs):
    """ Add to the pool of available configuration files for BIDSLayout.
    Args:
        kwargs: each kwarg should be a pair of config key name, and path

    Example: bids.layout.add_config_paths(my_config='/path/to/config')
    """

    for k, path in kwargs.items():
        if not os.path.exists(path):
            raise ValueError(
                'Configuration file "{}" does not exist'.format(k))
        if k in cf.get_option('config_paths'):
            raise ValueError('Configuration {!r} already exists'.format(k))

    kwargs.update(**cf.get_option('config_paths'))
    cf.set_option('config_paths', kwargs)


class BIDSFile(File):
    """ Represents a single BIDS file. """
    def __init__(self, filename, layout):
        super(BIDSFile, self).__init__(filename)
        self.layout = layout

    def __getattr__(self, attr):
        # Ensures backwards compatibility with old File_ namedtuple, which is
        # deprecated as of 0.7.
        if attr in self.entities:
            warnings.warn("Accessing entities as attributes is deprecated as "
                          "of 0.7. Please use the .entities dictionary instead"
                          " (i.e., .entities['%s'] instead of .%s."
                          % (attr, attr))
            return self.entities[attr]
        raise AttributeError("%s object has no attribute named %r" %
                             (self.__class__.__name__, attr))

    def __repr__(self):
        source = ''
        if self.layout.sources:
            source = ", root='{}'".format(os.path.basename(self.layout.root))
        return "<BIDSFile filename='{}'{}>".format(
            os.path.relpath(self.path, start=self.layout.root), source)

    @property
    def image(self):
        """ Return the associated image file (if it exists) as a NiBabel object.
        """
        try:
            import nibabel as nb
            return nb.load(self.path)
        except Exception:
            return None

    @property
    def metadata(self):
        """ Return all associated metadata. """
        return self.layout.get_metadata(self.path)


class BIDSLayout(Layout):
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
        include (str, list): String or list of strings specifying which of the
            directories that are by default excluded from indexing should be
            included. The default exclusion list is ['code', 'stimuli',
            'sourcedata', 'models'].
        absolute_paths (bool): If True, queries always return absolute paths.
            If False, queries return relative paths, unless the root argument
            was left empty (in which case the root defaults to the file system
            root).
        derivatives (bool, str, list): Specificies whether and/or which
            derivatives to to index. If True, all pipelines found in the
            derivatives/ subdirectory will be indexed. If a str or list, gives
            the paths to one or more derivatives directories to index. If False
            or None, the derivatives/ directory is ignored during indexing, and
            derivatives will have to be added manually via add_derivatives().
        config (str, list): Optional name(s) of configuration file(s) to use.
            By default (None), uses 'bids'.
        sources (BIDLayout, list): Optional BIDSLayout(s) from which the
            current BIDSLayout is derived.
        kwargs: Optional keyword arguments to pass onto the Layout initializer
            in grabbit.
    """

    def __init__(self, root, validate=True, index_associated=True,
                 include=None, absolute_paths=True, derivatives=False,
                 config=None, sources=None, **kwargs):

        self.validator = BIDSValidator(index_associated=index_associated)
        self.validate = validate
        self.metadata_index = MetadataIndex(self)
        self.derivatives = {}
        self.sources = listify(sources)

        # Validate arguments
        if not isinstance(root, six.string_types):
            raise ValueError("root argument must be a string specifying the"
                             " directory containing the BIDS dataset.")
        if not os.path.exists(root):
            raise ValueError("BIDS root does not exist: %s" % root)

        self.root = root

        target = os.path.join(self.root, 'dataset_description.json')
        if not os.path.exists(target):
            if validate is True:
                raise ValueError(
                    "'dataset_description.json' is missing from project root."
                    " Every valid BIDS dataset must have this file.")
            else:
                self.description = None
        else:
            with open(target, 'r', encoding='utf-8') as desc_fd:
                self.description = json.load(desc_fd)
            if validate is True:
                for k in ['Name', 'BIDSVersion']:
                    if k not in self.description:
                        raise ValueError("Mandatory '%s' field missing from "
                                         "dataset_description.json." % k)

        # Determine which subdirectories to exclude from indexing
        excludes = {"code", "stimuli", "sourcedata", "models", "derivatives"}
        if include is not None:
            include = listify(include)
            if "derivatives" in include:
                raise ValueError("Do not pass 'derivatives' in the include "
                                 "list. To index derivatives, either set "
                                 "derivatives=True, or use add_derivatives().")
            excludes -= set([d.strip(os.path.sep) for d in include])
        self._exclude_dirs = list(excludes)

        # Set up path and config for grabbit
        if config is None:
            config = 'bids'
        config_paths = get_option('config_paths')
        path = (root, [config_paths[c] for c in listify(config)])

        # Initialize grabbit Layout
        super(BIDSLayout, self).__init__(path, root=self.root,
                                         dynamic_getters=True,
                                         absolute_paths=absolute_paths,
                                         **kwargs)

        # Add derivatives if any are found
        self.derivatives = {}
        if derivatives:
            if derivatives is True:
                derivatives = os.path.join(root, 'derivatives')
            self.add_derivatives(
                derivatives, validate=validate,
                index_associated=index_associated, include=include,
                absolute_paths=absolute_paths, derivatives=None, config=None,
                sources=self, **kwargs)

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

        local_entities = set(ent.name for ent in self.entities.values())
        for deriv in deriv_dirs:
            dd = os.path.join(deriv, 'dataset_description.json')
            with open(dd, 'r', encoding='utf-8') as ddfd:
                description = json.load(ddfd)
            pipeline_name = description.get(
                'PipelineDescription', {}).get('Name', None)
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

            # Propagate derivative entities into top-level dynamic getters
            deriv_entities = set(
                ent.name
                for ent in self.derivatives[pipeline_name].entities.values())
            for deriv_ent in deriv_entities - local_entities:
                local_entities.add(deriv_ent)
                getter = 'get_' + inflect.engine().plural(deriv_ent)
                if not hasattr(self, getter):
                    func = partial(
                        self.get, target=deriv_ent, return_type='id')
                    setattr(self, getter, func)

    def to_df(self, **kwargs):
        """
        Return information for all Files tracked in the Layout as a pandas
        DataFrame.

        Args:
            kwargs: Optional keyword arguments passed on to get(). This allows
                one to easily select only a subset of files for export.
        Returns:
            A pandas DataFrame, where each row is a file, and each column is
                a tracked entity. NaNs are injected whenever a file has no
                value for a given attribute.
        """
        return self.as_data_frame(**kwargs)

    def __repr__(self):
        n_sessions = len([session for isub in self.get_subjects()
                          for session in self.get_sessions(subject=isub)])
        n_runs = len([run for isub in self.get_subjects()
                      for run in self.get_runs(subject=isub)])
        n_subjects = len(self.get_subjects())
        root = self.root[-30:]
        s = ("BIDS Layout: ...{} | Subjects: {} | Sessions: {} | "
             "Runs: {}".format(root, n_subjects, n_sessions, n_runs))
        return s

    def _validate_dir(self, d):
        # Callback from grabbit. Exclude special directories like derivatives/
        # and code/ from indexing unless they were explicitly included at
        # initialization.
        no_root = os.path.relpath(d, self.root).split(os.path.sep)[0]
        if no_root in self._exclude_dirs:
            check_paths = set(self._paths_to_index) - {self.root}
            if not any([d.startswith(p) for p in check_paths]):
                return False
        return True

    def _validate_file(self, f):
        # Callback from grabbit. Files are excluded from indexing if validation
        # is enabled and fails (i.e., file is not a valid BIDS file).
        if not self.validate:
            return True

        # For derivatives, we need to cheat a bit and construct a fake
        # derivatives path--prepend 'derivatives' and the pipeline name
        to_check = os.path.relpath(f, self.root)
        if 'derivatives' in self.domains:
            to_check = os.path.join(
                'derivatives', self.description['PipelineDescription']['Name'],
                to_check)

        sep = os.path.sep
        if to_check[:len(sep)] != sep:
            to_check = sep + to_check

        return self.validator.is_bids(to_check)

    def _get_nearest_helper(self, path, extension, suffix=None, **kwargs):
        """ Helper function for grabbit get_nearest """
        path = os.path.abspath(path)

        if not suffix:
            f = self.get_file(path)
            if 'suffix' not in f.entities:
                raise ValueError(
                    "File '%s' does not have a valid suffix, most "
                    "likely because it is not a valid BIDS file." % path
                )
            suffix = f.entities['suffix']

        tmp = self.get_nearest(
            path, extensions=extension, all_=True, suffix=suffix,
            ignore_strict_entities=['suffix'], **kwargs)

        if len(tmp):
            return tmp
        else:
            return None

    def get(self, return_type='object', target=None, extensions=None,
            derivatives=True, regex_search=None, defined_fields=None,
            domains=None, **kwargs):
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
                Files with any other extensions will be excluded.
            derivatives (bool, str, list): Whether/how to search associated
                BIDS-Derivatives datasets. If True (default), all available
                derivatives are searched. If a str or list, must be the name(s)
                of the derivatives to search (as defined in the
                PipelineDescription.Name field in dataset_description.json).
            regex_search (bool or None): Whether to require exact matching
                (False) or regex search (True) when comparing the query string
                to each entity. If None (default), uses the value found in
                self.
            defined_fields (list): Optional list of names of metadata fields
                that must be defined in JSON sidecars in order to consider the
                file a match, but which don't need to match any particular
                value.
            domains (str, list): Domain(s) to search in. Valid values are
                'bids' and 'derivatives'.
            kwargs (dict): Any optional key/values to filter the entities on.
                Keys are entity names, values are regexes to filter on. For
                example, passing filter={ 'subject': 'sub-[12]'} would return
                only files that match the first two subjects.

        Returns:
            A list of BIDSFile (default) or other objects
            (see return_type for details).
        """

        # Warn users still expecting 0.6 behavior
        if 'type' in kwargs:
            raise ValueError("As of pybids 0.7.0, the 'type' argument has been"
                             " replaced with 'suffix'.")

        if derivatives is True:
            derivatives = list(self.derivatives.keys())
        elif derivatives:
            derivatives = listify(derivatives)

        # Separate entity kwargs from metadata kwargs
        ent_kwargs, md_kwargs = {}, {}

        all_ents = self.get_domain_entities()
        if derivatives:
            for deriv in derivatives:
                deriv_ents = self.derivatives[deriv].get_domain_entities()
                all_ents.update(deriv_ents)

        for k, v in kwargs.items():
            if k in all_ents:
                ent_kwargs[k] = v
            else:
                md_kwargs[k] = v

        # Provide some suggestions if target is specified and invalid.
        if target is not None and target not in all_ents:
            import difflib
            potential = list(all_ents.keys())
            suggestions = difflib.get_close_matches(target, potential)
            if suggestions:
                message = "Did you mean one of: {}?".format(suggestions)
            else:
                message = "Valid targets are: {}".format(potential)
            raise ValueError(("Unknown target '{}'. " + message)
                             .format(target))

        all_results = []

        # Get entity-based search results using the superclass's get()
        result = []
        result = super(
            BIDSLayout, self).get(return_type, target=target,
                                  extensions=extensions, domains=None,
                                  regex_search=regex_search, **ent_kwargs)

        # Search the metadata if needed
        if return_type not in {'dir', 'id'}:

            if md_kwargs:
                if return_type.startswith('obj'):
                    result = [f.path for f in result]

                result = self.metadata_index.search(result, defined_fields,
                                                    **md_kwargs)

                if return_type.startswith('obj'):
                    result = [self.files[f] for f in result]

        all_results.append(result)

        # Add results from derivatives
        if derivatives:
            for deriv in derivatives:
                deriv = self.derivatives[deriv]
                deriv_res = deriv.get(return_type, target, extensions, None,
                                      regex_search, **ent_kwargs)
                all_results.append(deriv_res)

        # Flatten results
        result = list(chain(*all_results))
        if return_type in ['dir', 'id']:
            result = list(set(result))

        return result

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

        # For querying efficiency, store metadata in the MetadataIndex cache
        self.metadata_index.index_file(path)

        if include_entities:
            f = self.get_file(os.path.abspath(path))
            entities = f.entities
            results = entities
        else:
            results = {}

        results.update(self.metadata_index.file_index[path])
        return results

    def get_bvec(self, path, **kwargs):
        """Get bvec file for passed path."""
        tmp = self._get_nearest_helper(path, 'bvec', suffix='dwi', **kwargs)[0]
        if isinstance(tmp, list):
            return tmp[0]
        else:
            return tmp

    def get_bval(self, path, **kwargs):
        """Get bval file for passed path."""
        tmp = self._get_nearest_helper(path, 'bval', suffix='dwi', **kwargs)[0]
        if isinstance(tmp, list):
            return tmp[0]
        else:
            return tmp

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
        sub = os.path.split(path)[1].split("_")[0].split("sub-")[1]
        fieldmap_set = []
        suffix = '(phase1|phasediff|epi|fieldmap)'
        files = self.get(subject=sub, suffix=suffix,
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
        selectors['suffix'] = 'bold'
        selectors['datatype'] = 'func'
        images = self.get(extensions=['.nii', '.nii.gz'], derivatives=derivatives,
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

    def _make_file_object(self, root, f):
        # Override grabbit's File with a BIDSFile.
        return BIDSFile(os.path.join(root, f), self)

    def get_file(self, filename, derivatives=True):
        ''' Returns the BIDSFile object with the specified path.

        Args:
            filename (str): The path of the file to retrieve.
            derivatives (bool: If True, checks all associated derivative
                datasets as well.

        Returns: A BIDSFile.
        '''
        layouts = [self]
        if derivatives:
            layouts += self.derivatives.values()
        for ly in layouts:
            if filename in ly.files:
                return ly.files[filename]
        return None


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
        potential_jsons = self.layout._get_nearest_helper(path, '.json',
                                                          **kwargs)

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
