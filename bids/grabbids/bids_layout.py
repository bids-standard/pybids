import os
import re
import json

from os.path import dirname
from os.path import realpath
from os.path import join as pathjoin
from os.path import basename

from grabbit import Layout

__all__ = ['BIDSLayout']


class BIDSLayout(Layout):
    def __init__(self, path, config=None, **kwargs):
        if config is None:
            root = dirname(realpath(__file__))
            config = pathjoin(root, 'config', 'bids.json')
        super(BIDSLayout, self).__init__(path, config,
                                         dynamic_getters=True, **kwargs)

    def _validate_file(self, f):
        # Return False to exclude a file from indexing. This should call
        # some kind of validation regex.
        return True

    def get_metadata(self, path, **kwargs):
        path = realpath(path)

        if path not in self.files:
            raise ValueError("File '%s' could not be found in the current BIDS"
                             " project." % path)

        # Constrain the search to .json files with the same type as target
        type_ = self.files[path].entities['type']

        potentialJSONs = self.get_nearest(path, extensions='.json', all_=True,
                                          type=type_, **kwargs)

        merged_param_dict = {}
        for json_file_path in potentialJSONs:
            if os.path.exists(json_file_path):
                param_dict = json.load(open(json_file_path, "r"))
                merged_param_dict.update(param_dict)

        return merged_param_dict

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

    def find_match(self, target, source=None):

        # Try to take the easy way out
        if source is not None:
            _target = source.split('.')[0] + '.' + target
            if os.path.exists(_target):
                return target

        if target in list(self.entities.keys()):
            candidates = list(self.entities[target].files.keys())
        else:
            candidates = []

            for root, directories, filenames in os.walk(self.root):
                for f in filenames:
                    if re.search(target + '$', f):
                        if os.path.sep == "\\":
                            f = f.replace("\\", "\\\\")
                        candidates.append(f)

        if source is None:
            return candidates

        # Walk up the file hierarchy from source, find first match
        if not os.path.exists(source):
            raise OSError("The file '%s' doesn't exist." % source)
        elif not source.startswith(self.root):
            raise ValueError("The file '%s' is not contained "
                             "within the current project "
                             "directory (%s)." % (source, self.root))
        rel = os.path.relpath(dirname(source), self.root)
        sep = os.path.sep
        chunks = rel.split(sep)
        n_chunks = len(chunks)
        for i in range(n_chunks, -1, -1):
            path = pathjoin(self.root, *chunks[:i])
            patt = path + '\%s[^\%s]+$' % (sep, sep)
            if sep == "\\":
                patt = path + '\\[^\\]+$'
                patt = patt.replace("\\", "\\\\")
            matches = [x for x in candidates if re.search(patt, x)]
            if matches:
                if len(matches) == 1:
                    return matches[0]
                else:
                    raise ValueError("Ambiguous target: more than one "
                                     "candidate file found in "
                                     "directory '%s'." % path)
        return None
