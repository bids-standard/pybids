from grabbit import Layout
import os
import re
import json
import itertools

__all__ = ['BIDSLayout']


class BIDSLayout(Layout):
    def __init__(self, path, config=None):
        if config is None:
            root = os.path.dirname(os.path.realpath(__file__))
            config = os.path.join(root, 'config', 'bids.json')
        super(BIDSLayout, self).__init__(path, config)

    def get_metadata(self, path):
        sidecarJSON = path.replace(".nii.gz", ".json").replace(".nii", ".json")
        pathComponents = os.path.split(sidecarJSON)
        filenameComponents = pathComponents[-1].split("_")
        ses = None
        suffix = filenameComponents[-1]

        sub = filenameComponents[0]
        keyword_components = filenameComponents[1:-1]
        if filenameComponents[1][:3] == "ses":
            ses = filenameComponents[1]
            keyword_components = filenameComponents[2:-1]

        potentialJSONs = []
        for k in range(len(keyword_components) + 1):
            print(k)
            for components in itertools.combinations(keyword_components, k):
                print(components)
                potentialJSONs.append(os.path.join(self.root,
                                                   "_".join(components +
                                                            (suffix,))))

        for k in range(len(keyword_components) + 1):
            for components in itertools.combinations(keyword_components, k):
                potentialJSONs.append(os.path.join(self.root, sub,
                                                   "_".join((sub,) +
                                                            components +
                                                            (suffix,))))

        if ses:
            for k in range(len(keyword_components) + 1):
                for components in itertools.combinations(keyword_components,
                                                         k):
                    potentialJSONs.append(os.path.join(self.root, sub, ses,
                                                       "_".join((sub, ses) +
                                                                components +
                                                                (suffix,))))

        merged_param_dict = {}
        for json_file_path in potentialJSONs:
            if os.path.exists(json_file_path):
                param_dict = json.load(open(json_file_path, "r"))
                merged_param_dict.update(param_dict)

        return merged_param_dict

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
        rel = os.path.relpath(os.path.dirname(source), self.root)
        sep = os.path.sep
        chunks = rel.split(sep)
        n_chunks = len(chunks)
        for i in range(n_chunks, -1, -1):
            path = os.path.join(self.root, *chunks[:i])
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
