import os
import json
import warnings
from io import open
from .validation import BIDSValidator
from grabbit import Layout, File
from grabbit.external import six
from grabbit.utils import listify
import nibabel as nb
from collections import defaultdict
from functools import reduce
import re


try:
    from os.path import commonpath
except ImportError:
    def commonpath(paths):
        prefix = os.path.commonprefix(paths)
        if not os.path.isdir(prefix):
            prefix = os.path.dirname(prefix)
        return prefix


__all__ = ['BIDSLayout']


class BIDSFile(File):
    """ Represents a single BIDS file. """
    def __init__(self, filename, layout):
        super(BIDSFile, self).__init__(filename)
        self.layout = layout

    @property
    def image(self):
        """ Return the associated image file (if it exists) as a NiBabel object.
        """
        try:
            return nb.load(self.path)
        except Exception as e:
            return None

    @property
    def metadata(self):
        """ Return all associated metadata. """
        return self.layout.get_metadata(self.path)


class BIDSLayout(Layout):
    """ Layout class representing an entire BIDS project.

    Args:
        paths (str, list): The path(s) where project files are located.
                Must be one of:

                - A path to a directory containing files to index
                - A list of paths to directories to be indexed
                - A list of 2-tuples where each tuple encodes a mapping from
                  directories to domains. The first element is a string or
                  list giving the paths to one or more directories to index.
                  The second element specifies which domains to apply to the
                  specified files, and can be one of:
                    * A string giving the name of a built-in config
                    * A string giving the path to a JSON config file
                    * A dictionary containing config information
                    * A list of any combination of strings or dicts

            At present, built-in domains include 'bids' and 'derivatives'.

        root (str): The root directory of the BIDS project. All other paths
            will be set relative to this if absolute_paths is False. If None,
            filesystem root ('/') is used.
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
            included. The default exclusion list is ['derivatives', 'code',
            'stimuli', 'sourcedata', 'models']. Note that if directories are
            specified in the paths argument, they do not also need to be
            included here again.
        absolute_paths (bool): If True, queries always return absolute paths.
            If False, queries return relative paths, unless the root argument
            was left empty (in which case the root defaults to the file system
            root).
        kwargs: Optional keyword arguments to pass onto the Layout initializer
            in grabbit.
    """

    def __init__(self, paths, root=None, validate=True, index_associated=True,
                 include=None, absolute_paths=True, index_metadata=False,
                 **kwargs):

        self.validator = BIDSValidator(index_associated=index_associated)
        self.validate = validate
        self.metadata_index = None

        # Determine which configs to load
        conf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'config', '%s.json')
        all_confs = ['bids', 'derivatives']

        def map_conf(x):
            if isinstance(x, six.string_types) and x in all_confs:
                return conf_path % x
            return x

        paths = listify(paths, ignore=list)

        for i, p in enumerate(paths):
            if isinstance(p, six.string_types):
                paths[i] = (p, conf_path % 'bids')
            elif isinstance(p, tuple):
                conf_names = listify(p[1])
                # All 'derivatives' files are also 'bids' files. This is hacky
                # and should be replaced with something more principled.
                if 'derivatives' in conf_names:
                    conf_names = list(set(conf_names) | {'bids'})
                # Map each built-in config name to the JSON file
                doms = [map_conf(d) for d in conf_names]
                paths[i] = (p[0], doms)

        # Determine which subdirectories to exclude from indexing
        excludes = {"derivatives", "code", "stimuli", "sourcedata", "models"}
        if include is not None:
            excludes -= set([d.strip(os.path.sep) for d in include])
        # Calculate directories to exclude
        self._exclude_dirs = []
        for (p, _) in paths:
            self._exclude_dirs += [os.path.join(p, e) for e in excludes]

        # Set root to longest valid common parent if it isn't explicitly set
        if root is None:
            abs_paths = [os.path.abspath(p[0]) for p in paths]
            root = commonpath(abs_paths)
            if not root:
                raise ValueError("One or more invalid paths passed; could not "
                                 "find a common parent directory of %s. Either"
                                 " make sure the paths are correct, or "
                                 "explicitly set the root using the 'root' "
                                 "argument." % abs_paths)

        self.root = root

        target = os.path.join(self.root, 'dataset_description.json')
        if not os.path.exists(target):
            warnings.warn("'dataset_description.json' file is missing from "
                          "project root. You may want to set the root path to "
                          "a valid BIDS project.")
            self.description = None
        else:
            self.description = json.load(open(target, 'r'))
            for k in ['Name', 'BIDSVersion']:
                if k not in self.description:
                    raise ValueError("Mandatory '%s' field missing from "
                                     "dataset_description.json." % k)

        super(BIDSLayout, self).__init__(paths, root=root,
                                         dynamic_getters=True,
                                         absolute_paths=absolute_paths,
                                         **kwargs)

        if index_metadata:
            self.build_metadata_index()

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
        # initialization in either the include or paths arguments
        if any([commonpath([os.path.abspath(c)]) == commonpath([os.path.abspath(c), os.path.abspath(d)])
                for c in self._exclude_dirs]):
                return False
        return True

    def _validate_file(self, f):
        # Callback from grabbit. Files are excluded from indexing if validation
        # is enabled and fails (i.e., file is not a valid BIDS file).
        if not self.validate:
            return True
        to_check = f.split(os.path.abspath(self.root), 1)[1]

        sep = os.path.sep
        if to_check[:len(sep)] != sep:
            to_check = sep + to_check
        else:
            None

        return self.validator.is_bids(to_check)

    def _get_nearest_helper(self, path, extension, suffix=None, **kwargs):
        """ Helper function for grabbit get_nearest """
        path = os.path.abspath(path)

        if not suffix:
            if 'suffix' not in self.files[path].entities:
                raise ValueError(
                    "File '%s' does not have a valid suffix, most "
                    "likely because it is not a valid BIDS file." % path
                )
            suffix = self.files[path].entities['suffix']

        tmp = self.get_nearest(path, extensions=extension, all_=True,
                               suffix=suffix, ignore_strict_entities=['suffix'],
                               **kwargs)

        if len(tmp):
            return tmp
        else:
            return None

    def get_metadata(self, path, include_entities=False, **kwargs):
        """Return metadata found in JSON sidecars for the specified file.

        Args:
            path (str): Path to the file to get metadata for.
            include_entities (bool): If True, all available entities extracted
                from the filename (rather than JSON sidecars) are included in
                the returned metadata dictionary.
            kwargs (dict): Optional keyword arguments to pass onto
                get_nearest().
        Notes:
            A dictionary containing metadata extracted from all matching .json
            files is returned. In cases where the same key is found in multiple
            files, the values in files closer to the input filename will take
            precedence, per the inheritance rules in the BIDS specification.

        """
        if include_entities:
            entities = self.files[os.path.abspath(path)].entities
            merged_param_dict = entities
        else:
            merged_param_dict = {}

        potential_jsons = self._get_nearest_helper(path, '.json', **kwargs)

        if potential_jsons is None:
            return merged_param_dict

        for json_file_path in reversed(potential_jsons):
            if os.path.exists(json_file_path):
                param_dict = json.load(open(json_file_path, "r",
                                            encoding='utf-8'))
                merged_param_dict.update(param_dict)

        return merged_param_dict

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
            metadata = self.get_metadata(file.filename)
            if metadata and "IntendedFor" in metadata.keys():
                intended_for = listify(metadata["IntendedFor"])
                if any([path.endswith(_suff) for _suff in intended_for]):
                    cur_fieldmap = {}
                    if file.suffix == "phasediff":
                        cur_fieldmap = {"phasediff": file.filename,
                                        "magnitude1": file.filename.replace(
                                            "phasediff", "magnitude1"),
                                        "suffix": "phasediff"}
                        magnitude2 = file.filename.replace(
                            "phasediff", "magnitude2")
                        if os.path.isfile(magnitude2):
                            cur_fieldmap['magnitude2'] = magnitude2
                    elif file.suffix == "phase1":
                        cur_fieldmap["phase1"] = file.filename
                        cur_fieldmap["magnitude1"] = \
                            file.filename.replace("phase1", "magnitude1")
                        cur_fieldmap["phase2"] = \
                            file.filename.replace("phase1", "phase2")
                        cur_fieldmap["magnitude2"] = \
                            file.filename.replace("phase1", "magnitude2")
                        cur_fieldmap["suffix"] = "phase"
                    elif file.suffix == "epi":
                        cur_fieldmap["epi"] = file.filename
                        cur_fieldmap["suffix"] = "epi"
                    elif file.suffix == "fieldmap":
                        cur_fieldmap["fieldmap"] = file.filename
                        cur_fieldmap["magnitude"] = \
                            file.filename.replace("fieldmap", "magnitude")
                        cur_fieldmap["suffix"] = "fieldmap"
                    fieldmap_set.append(cur_fieldmap)
        return fieldmap_set

    def get_collections(self, level, types=None, variables=None, merge=False,
                        sampling_rate=None, skip_empty=False, **kwargs):
        """Return one or more variable Collections in the BIDS project.

        Args:
            level (str): The level of analysis to return variables for. Must be
                one of 'run', 'session', 'subject', or 'dataset'.
            types (str, list): Types of variables to retrieve. All valid values
            reflect the filename stipulated in the BIDS spec for each kind of
            variable. Valid values include: 'events', 'physio', 'stim',
            'scans', 'participants', 'sessions', and 'confounds'.
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

    def build_metadata_index(self, regex_search=False, preserve_dtypes=True):
        """Build a metadata index for the current BIDSLayout.

        Args:
            regex_search (bool): If True, matching is performed using regex.
            preserve_dtypes (bool): If True (default), the dtype of the
                original metadata value is preserved when matching (so 6 won't
                match '6'); if False, all matching will be done after
                converting values to strings. (Note that if regex_search=True,
                preserve_dtypes will be automatically set to True as well).
        """
        self.metadata_index = MetadataIndex(self, regex_search,
                                            preserve_dtypes)

    def search_metadata(self, files=None, regex_search=None,
                        defined_fields=None, **kwargs):
        """Search files based on metadata fields.

        Args:
            files (list): Optional list of names of files to search. If None,
                all files in the layout are scanned.
            regex_search (bool): If True, matching is performed using regex.
                If False, no regex is used. If None, the value is taken from
                the current instance.
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
        if self.metadata_index is None:
            raise ValueError("No metadata index was found. Before you can "
                             "search on file metadata, you need to either "
                             "pass index_metadata=True when initializing "
                             "the BIDSLayout, or explicitly call "
                             "build_metadata_index() on the BIDSLayout.")
        return self.metadata_index.search(files, regex_search, defined_fields,
                                          **kwargs)


class MetadataIndex(object):
    """A simple dict-based index for key/value pairs in JSON metadata.

    Args:
        layout (BIDSLayout): The BIDSLayout instance to index.
        regex_search (bool): If True, matching is performed using regex.
        preserve_dtypes (bool): If True (default), the dtype of the original
            metadata value is preserved when matching (so 6 won't match '6');
            if False, all matching will be done after converting values to
            strings. (Note that if regex_search=True, preserve_dtypes will be
            automatically set to True as well).
    """

    def __init__(self, layout, regex_search=False, preserve_dtypes=True):
        self.regex_search = regex_search
        self.preserve_dtypes = preserve_dtypes
        self.key_index = defaultdict(dict)
        self.file_index = defaultdict(dict)
        for f_name, f in layout.files.items():
            if f_name.endswith('.json'):
                continue
            md = f.metadata
            for md_key, md_val in md.items():
                if not preserve_dtypes:
                    md_val = str(md_val)
                self.key_index[md_key][f_name] = md_val
                self.file_index[f_name][md_key] = md_val

    def search(self, files=None, regex_search=None, defined_fields=None,
               **kwargs):
        """Search files in the layout by metadata fields.

        Args:
            files (list): Optional list of names of files to search. If None,
                all files in the layout are scanned.
            regex_search (bool): If True, matching is performed using regex.
                If False, no regex is used. If None, the value is taken from
                the current instance.
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

        if regex_search is None:
            regex_search = self.regex_search

        if defined_fields is None:
            defined_fields = []

        all_keys = set(defined_fields) | set(kwargs.keys())
        if not all_keys:
            raise ValueError("At least one field to search on must be passed.")

        # Get file intersection of all kwargs keys--this is fast
        filesets = [set(self.key_index[k]) for k in all_keys]
        matches = reduce(lambda x, y: x & y, filesets)

        if files is not None:
            matches &= set(files)

        if not matches:
            return []

        def check_matches(f, key, val):
            if regex_search:
                return re.search(str(self.file_index[f][key]), val) is not None
            else:
                return val == self.file_index[f][key]

        # Serially check matches against each pattern, with early termination
        for k, val in kwargs.items():
            if not self.preserve_dtypes:
                val = str(val)
            matches = list(filter(lambda x: check_matches(x, k, val), matches))
            if not matches:
                return []

        return matches
