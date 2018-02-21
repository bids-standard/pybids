import os
import json
import collections
import warnings
from io import open

from os.path import dirname
from os.path import abspath
from os.path import join as pathjoin

from .bids_validator import BIDSValidator
from .utils import _merge_event_files
from grabbit import Layout, File
from grabbit.external import six


__all__ = ['BIDSLayout']


class BIDSLayout(Layout):
    ''' Layout class representing an entire BIDS project.

    Args:
        path (str): The path specifying the root directory of the BIDS project.
        config (list, str): A string or list of string specifying which config
            files to load. Each element in the list (or the value, if a string)
            must be either a path to a JSON domain configuration file (see
            grabbit docs for explanation), or the name of a built-in domain.
            At present, built-in domains include 'bids' and 'derivatives'.
        validate (bool): If True, all files are checked for BIDS compliance
            when first indexed, and non-compliant files are ignored. Note that
            the validator is experimental and may fail to perfectly detect
            compliance.
        index_associated (bool): Argument passed onto the BIDSValidator;
            ignored if validate = False.
        kwargs: Optional keyword arguments to pass onto the Layout initializer
            in grabbit.
    '''

    def __init__(self, path, config=None, validate=False,
                 index_associated=True, **kwargs):
        self.validator = BIDSValidator(index_associated=index_associated)
        self.validate = validate

        # Determine which configs to load
        conf_path = pathjoin(dirname(abspath(__file__)), 'config', '%s.json')
        _all_doms = ['bids', 'derivatives']
        if config is None:
            config = ['bids']

        # If 'bids' isn't in the list, the user probably made a mistake...
        if 'bids' not in config:
            warnings.warn("The core BIDS configuration was not included in the"
                          " config list. If you override the default value for"
                          " config, you probably want to make sure 'bids' is "
                          "included in the list of values.")

        config = [conf_path % d if d in _all_doms else d for d in config]
        config = [json.load(open(c, 'r')) for c in config]

        # A bit hacky, but if derivatives are included, we need to make sure
        # the derivatives directory isn't listed in excludes
        if any([c['name'] == 'derivatives' for c in config]):
            bids = [c for c in config if c['name'] == 'bids'][0]
            bids['index']['exclude'].pop(0)

        super(BIDSLayout, self).__init__(path, config=config,
                                         dynamic_getters=True, **kwargs)

    def _validate_file(self, f):
        # If validate=True then checks files according to BIDS and
        # returns False if file doesn't fit BIDS specification
        if not self.validate:
            return True
        to_check = f.path
        to_check = to_check.split(os.path.abspath(self.root), maxsplit=1)[1]

        sep = os.path.sep
        if to_check[:len(sep)] != sep:
            to_check = sep + to_check
        else:
            None

        return self.validator.is_bids(to_check)

    def _get_nearest_helper(self, path, extension, type=None, **kwargs):
        """ Helper function for grabbit get_nearest """
        path = abspath(path)

        if not type:
            if 'type' not in self.files[path].entities:
                raise ValueError(
                    "File '%s' does not have a valid type definition, most "
                    "likely because it is not a valid BIDS file." % path
                )
            type = self.files[path].entities['type']

        tmp = self.get_nearest(path, extensions=extension, all_=True,
                               type=type, ignore_strict_entities=['type'],
                               **kwargs)

        if len(tmp):
            return tmp
        else:
            return None

    def get_metadata(self, path, include_entities=False, **kwargs):
        ''' Returns metadata found in JSON sidecars for the specified file.
        Args:
            path (str): Path to the file to get metadata for.
            kwargs (dict): Optional keyword arguments to pass onto
                get_nearest().
        Notes:
            A dictionary containing metadata extracted from all matching .json
            files is returned. In cases where the same key is found in multiple
            files, the values in files closer to the input filename will take
            precedence, per the inheritance rules in the BIDS specification.
        '''

        if include_entities:
            entities = self.files[abspath(path)].entities
            merged_param_dict = entities
        else:
            merged_param_dict = {}

        potentialJSONs = self._get_nearest_helper(path, '.json', **kwargs)

        if potentialJSONs is None:
            return merged_param_dict

        for json_file_path in reversed(potentialJSONs):
            if os.path.exists(json_file_path):
                param_dict = json.load(open(json_file_path, "r",
                                            encoding='utf-8'))
                merged_param_dict.update(param_dict)

        return merged_param_dict

    def get_bvec(self, path, **kwargs):
        tmp = self._get_nearest_helper(path, 'bvec', type='dwi', **kwargs)[0]
        if isinstance(tmp, list):
            return tmp[0]
        else:
            return tmp

    def get_bval(self, path, **kwargs):
        tmp = self._get_nearest_helper(path, 'bval', type='dwi', **kwargs)[0]
        if isinstance(tmp, list):
            return tmp[0]
        else:
            return tmp

    def get_events(self, path, return_type='file', derivatives='both',
                   **kwargs):
        """ For a given file in a BIDS project, finds corresponding event files
        and optionally returns merged dataframe containing all variables.

        Args:
            path (str): Path to a file to match to events.
            return_type (str): Type of output to return.
                'file' returns list of files,
                'df' merges events into a single DataFrame, giving precedence
                to events closer to the file.
            derivatives (str): How to handle derivative events.
                'ignore' - Ignore any event files outside of root directory.
                'only' - Only include event files from outside directories.
                'both' - Include both. Derivative events have precedence.
        Returns:
            List of file or merged Pandas dataframe.
        """

        path = abspath(path)

        # Get events in base Layout directory (ordered)
        root_events = self._get_nearest_helper(
            path, '.tsv', type='events', **kwargs) or []

        entities = self.files[path].entities.copy()

        if 'type' in entities:
            entities.pop('type')
        if 'modality' in entities and entities['modality'] == 'func':
            entities.pop('modality')

        entities.update(kwargs)

        # Get all events
        events = self.get(extensions='tsv', type='events',
                          return_type='file', **entities) or []

        deriv_events = list(set(events) - set(root_events))

        if derivatives == 'only':
            events = deriv_events
        elif derivatives == 'ignore':
            events = root_events
        else: # Combine with order
            events = deriv_events + root_events

        if return_type == 'df':
            events = _merge_event_files(events)
        elif not events:
            return None
        elif len(events) == 1:
            return events[0]
        return events

    def get_fieldmap(self, path, return_list=False):
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
        type_ = '(phase1|phasediff|epi|fieldmap)'
        files = self.get(subject=sub, type=type_, extensions=['nii.gz', 'nii'])
        for file in files:
            metadata = self.get_metadata(file.filename)
            if metadata and "IntendedFor" in metadata.keys():
                if isinstance(metadata["IntendedFor"], list):
                    intended_for = metadata["IntendedFor"]
                else:
                    intended_for = [metadata["IntendedFor"]]
                if any([path.endswith(suffix) for suffix in intended_for]):
                    cur_fieldmap = {}
                    if file.type == "phasediff":
                        cur_fieldmap = {"phasediff": file.filename,
                                        "magnitude1": file.filename.replace(
                                            "phasediff", "magnitude1"),
                                        "type": "phasediff"}
                        magnitude2 = file.filename.replace(
                            "phasediff", "magnitude2")
                        if os.path.isfile(magnitude2):
                            cur_fieldmap['magnitude2'] = magnitude2
                    elif file.type == "phase1":
                        cur_fieldmap["phase1"] = file.filename
                        cur_fieldmap["magnitude1"] = \
                            file.filename.replace("phase1", "magnitude1")
                        cur_fieldmap["phase2"] = \
                            file.filename.replace("phase1", "phase2")
                        cur_fieldmap["magnitude2"] = \
                            file.filename.replace("phase1", "magnitude2")
                        cur_fieldmap["type"] = "phase"
                    elif file.type == "epi":
                        cur_fieldmap["epi"] = file.filename
                        cur_fieldmap["type"] = "epi"
                    elif file.type == "fieldmap":
                        cur_fieldmap["fieldmap"] = file.filename
                        cur_fieldmap["magnitude"] = \
                            file.filename.replace("fieldmap", "magnitude")
                        cur_fieldmap["type"] = "fieldmap"
                    fieldmap_set.append(cur_fieldmap)
        return fieldmap_set

    def get_collections(self, level, types=None, variables=None, merge=False,
                        sampling_rate=None, **kwargs):
        from bids.variables import load_variables
        index = load_variables(self, types=types, levels=level, **kwargs)
        return index.get_collections(level, variables, merge,
                                     sampling_rate=sampling_rate)

    def parse_entities(self, filelike):
        if not isinstance(filelike, File):
            filelike = File(filelike)

        for ent in self.entities.values():
            ent.matches(filelike)

        return filelike.entities
