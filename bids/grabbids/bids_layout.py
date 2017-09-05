import os
import json

from os.path import dirname
from os.path import abspath
from os.path import join as pathjoin

from grabbit import Layout

__all__ = ['BIDSLayout']


class BIDSLayout(Layout):
    def __init__(self, path, config=None, **kwargs):
        if config is None:
            root = dirname(abspath(__file__))
            config = pathjoin(root, 'config', 'bids.json')
        super(BIDSLayout, self).__init__(path, config,
                                         dynamic_getters=True, **kwargs)
                                         
    def _validate_file(self, f):
        # Return False to exclude a file from indexing. This should call
        # some kind of validation regex.
        return super(BIDSLayout, self)._validate_file(f)

    def _get_nearest_helper(self, path, extension, type=None, **kwargs):
<<<<<<< HEAD
        """ Helper function for grabbit get_nearest """
=======
>>>>>>> upstream/master
        path = abspath(path)

        if path not in self.files:
            raise ValueError("File '%s' could not be found in the current BIDS"
                             " project." % path)

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
<<<<<<< HEAD
        """ Return metadata by merging matching JSON sidecars in order of
            distance from target file """
        potentialJSONs = self._get_nearest_helper(path, '.json', **kwargs)
=======

        potentialJSONs = self._get_nearest_helper(path, '.json', **kwargs)
        if not isinstance(potentialJSONs, list): return potentialJSONs
>>>>>>> upstream/master

        merged_param_dict = {}
        for json_file_path in reversed(potentialJSONs):
            if os.path.exists(json_file_path):
                param_dict = json.load(open(json_file_path, "r"))
                merged_param_dict.update(param_dict)

        return merged_param_dict

    def get_bvec(self, path, **kwargs):
<<<<<<< HEAD
        return self._get_nearest_helper(path, '.bvec', **kwargs)[0]

    def get_bval(self, path, **kwargs):
        return self._get_nearest_helper(path, '.bval', **kwargs)[0]

    def get_events(self, path, **kwargs):
        return self._get_nearest_helper(path, '.tsv', 'events', **kwargs)[0]
=======
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
>>>>>>> upstream/master

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
                                        "magnitude2": file.filename.replace(
                                            "phasediff", "magnitude2"),
                                        "type": "phasediff"}
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
