import os
import json

from io import open

from os.path import dirname
from os.path import abspath
from os.path import join as pathjoin

from .bids_validator import BIDSValidator
from grabbit import Layout

__all__ = ['BIDSLayout']


class BIDSLayout(Layout):

    def __init__(self, path, config=None, validate=False,
                 index_associated=True, **kwargs):
        self.validator = BIDSValidator(index_associated=index_associated)
        self.validate = validate
        if config is None:
            root = dirname(abspath(__file__))
            config = pathjoin(root, 'config', 'bids.json')
        super(BIDSLayout, self).__init__(path, config,
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

        if path not in self.files:
            raise ValueError(
                "File '%s' does not match any specification in the current "
                "BIDS project." % path)

        if not type:
            # Constrain the search to .json files with the same type as target
            type = self.files[path].entities['type']

        tmp = self.get_nearest(path, extensions=extension, all_=True,
                               type=type, ignore_strict_entities=['type'],
                               **kwargs)

        if len(tmp):
            return tmp
        else:
            return None

    def get_metadata(self, path, **kwargs):
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
        potentialJSONs = self._get_nearest_helper(path, '.json', **kwargs)

        if not isinstance(potentialJSONs, list):
            return potentialJSONs

        merged_param_dict = {}
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

    def get_events(self, path, **kwargs):
        tmp = self._get_nearest_helper(path, '.tsv', type='events', **kwargs)
        if isinstance(tmp, list):
            return tmp[0]
        else:
            return tmp

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
        for file in self.get(subject=sub, type=type_,
                             extensions=['nii.gz', 'nii']):
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
