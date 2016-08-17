from grabbit import Layout
from bids import grabbids
import os
import re

__all__ = ['BIDSLayout']


class BIDSLayout(Layout):

    def __init__(self, path, config=None):
        if config is None:
            root = os.path.dirname(grabbids.__file__)
            config = os.path.join(root, 'config', 'bids.json')
        super(BIDSLayout, self).__init__(path, config)

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
            matches = [x for x in candidates if re.search(patt, x)]
            if matches:
                if len(matches) == 1:
                    return matches[0]
                else:
                    raise ValueError("Ambiguous target: more than one "
                                     "candidate file found in "
                                     "directory '%s'." % path)
        return None
