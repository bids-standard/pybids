import os
import re
import json

from itertools import combinations
from os.path import dirname
from os.path import realpath
from os.path import join as pathjoin
from os.path import split as pathsplit

from grabbit import Layout

__all__ = ['BIDSLayout']


class BIDSLayout(Layout):
    def __init__(self, path, config=None, **kwargs):
        if config is None:
            root = dirname(realpath(__file__))
            config = pathjoin(root, 'config', 'bids.json')
        super(BIDSLayout, self).__init__(path, config,
                                         dynamic_getters=True, **kwargs)

    def get_metadata(self, path):
        sidecarJSON = path.replace(".nii.gz", ".json").replace(".nii", ".json")
        path_components = pathsplit(sidecarJSON)
        filename_components = path_components[-1].split("_")
        ses = None
        suffix = filename_components[-1]

        sub = filename_components[0]
        keyword_components = filename_components[1:-1]
        if filename_components[1][:3] == "ses":
            ses = filename_components[1]
            keyword_components = filename_components[2:-1]

        potentialJSONs = []
        for prefixes, midlayer, conditional in (  # Levels
                (tuple(), tuple(), True),        # top
                ((sub,), tuple(), True),         # subject
                ((sub, ), (pathsplit(path_components[-2])[-1],), True),
                ((sub, ses), tuple(), ses),  # session
                ((sub, ses), (pathsplit(path_components[-2])[-1],), ses)
        ):
            if not conditional:
                continue
            for k in range(len(keyword_components) + 1):
                for components in combinations(keyword_components, k):
                    potentialJSONs.append(
                        pathjoin(
                            self.root,
                            *(prefixes + midlayer +
                              ("_".join(prefixes + components + (suffix,)),))))

        merged_param_dict = {}
        for json_file_path in potentialJSONs:
            if os.path.exists(json_file_path):
                param_dict = json.load(open(json_file_path, "r"))
                merged_param_dict.update(param_dict)

        return merged_param_dict

    def get_fieldmap(self, path):
        sub = os.path.split(path)[1].split("_")[0].split("sub-")[1]
        fieldmap_set = {}
        for file in self.get(subject=sub,
                             type='(phase1|phase2|phasediff|epi|fieldmap)',
                             extensions=['nii.gz', 'nii']):
            metadata = self.get_metadata(file.filename)
            if metadata and "IntendedFor" in metadata.keys():
                if isinstance(metadata["IntendedFor"], list):
                    intended_for = metadata["IntendedFor"]
                else:
                    intended_for = [metadata["IntendedFor"]]
                if any([path.endswith(suffix) for suffix in intended_for]):
                    if file.type == "phasediff":
                        fieldmap_set = {"phasediff": file.filename,
                                        "magnitude1": file.filename.replace(
                                            "phasediff", "magnitude1"),
                                        "magnitude2": file.filename.replace(
                                            "phasediff", "magnitude2"),
                                        "type": "phasediff"}
                        break
                    elif file.type == "phase1":
                        fieldmap_set["phase1"] = file.filename
                        fieldmap_set["magnitude1"] = \
                            file.filename.replace("phase1", "magnitude1")
                        fieldmap_set["type"] = "phase"
                    elif file.type == "phase2":
                        fieldmap_set["phase2"] = file.filename
                        fieldmap_set["magnitude2"] = \
                            file.filename.replace("phase2", "magnitude2")
                        fieldmap_set["type"] = "phase"
                    elif file.type == "epi":
                        if "epi" not in fieldmap_set.keys():
                            fieldmap_set["epi"] = []
                        fieldmap_set["epi"].append(file.filename)
                        fieldmap_set["type"] = "epi"
                    elif file.type == "fieldmap":
                        fieldmap_set["fieldmap"] = file.filename
                        fieldmap_set["magnitude"] = \
                            file.filename.replace("fieldmap", "magnitude")
                        fieldmap_set["type"] = "fieldmap"
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
