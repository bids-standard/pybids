""" File-indexing functionality. """

import os
import re
import json
from keyword import iskeyword
import warnings
from copy import deepcopy
from collections import defaultdict

from .writing import build_path, write_contents_to_file
from .models import Config, BIDSFile, Entity, Tag
from ..utils import listify, check_path_matches_patterns
from ..config import get_option
from ..external import six


def _extract_entities(bidsfile, entities):
    match_vals = {}
    for e in entities.values():
        m = e.match_file(bidsfile)
        if m is None and e.mandatory:
            break
        if m is not None:
            match_vals[e.name] = (e, m)
    return match_vals


def index_layout(layout, config, force_index=None, index_metadata=True):

    session = layout.session
    root_path = layout.root

    # Track ALL entities we've seen in file names or metadata, starting with
    # those found in the configs
    all_entities = {}
    for c in config:
        all_entities.update(c.entities)

    # For metadata indexing, we build up a store of all JSON data as we walk
    # the file tree. It looks like: { dirname: { suffix: (entities, payload)}}
    json_data = {}

    def _index_file(f, dirpath, entities):
        ''' Create DB record for file and its tags. '''
        abs_fn = os.path.join(dirpath, f)

        # Skip files that fail validation, unless forcibly indexing
        if not force_index and not layout._validate_file(abs_fn):
            return None, entities

        bf = BIDSFile(abs_fn)
        session.add(bf)

        # Extract entity values
        match_vals = {}
        for e in entities.values():
            m = e.match_file(bf)
            if m is None and e.mandatory:
                break
            if m is not None:
                match_vals[e.name] = (e, m)

        # Create Entity <=> BIDSFile mappings
        if match_vals:
            for _, (ent, val) in match_vals.items():
                tag = Tag(bf, ent, str(val), ent._dtype)
                session.add(tag)

        session.commit()

        return bf, match_vals

    def index_dir(path, config, parent=None, force_index=False):

        abs_path = os.path.join(root_path, path)
        config = list(config)     # Shallow copy

        # Check for additional config file in directory
        layout_file = layout.config_filename
        config_file = os.path.join(abs_path, layout_file)
        if os.path.exists(config_file):
            cfg = Config.load(config_file, session=session)
            config.append(cfg)

        # Track which entities are valid in filenames for this directory
        config_entities = {}
        for c in config:
            config_entities.update(c.entities)

        if index_metadata:
            json_data[path] = defaultdict(list)

        for (dirpath, dirnames, filenames) in os.walk(path):

            # If layout configuration file exists, delete it
            if layout.config_filename in filenames:
                filenames.remove(layout.config_filename)

            # Process JSON files first if we're indexing metadata
            if index_metadata:
                sidecars = []

                for f in filenames:
                    if f.endswith('.json'):
                        filenames.remove(f)
                        bf, file_ents = _index_file(f, dirpath, config_entities)
                        if bf is None:
                            continue
                        suffix = file_ents.pop('suffix', None)
                        if suffix is not None:
                            with open(os.path.join(dirpath, f), 'r') as handle:
                                payload = json.load(handle)
                                if payload:
                                    to_store = (set(file_ents.keys()), payload)
                                    json_data[path][suffix].append(to_store)


            for f in filenames:

                bf, file_ents = _index_file(f, dirpath, config_entities)

                if bf is None:
                    continue

                # Add metadata
                if index_metadata:

                    suffix = file_ents.pop('suffix', None)
                    file_ents = set(file_ents.keys())

                    if suffix is None:
                        continue

                    # Extract metadata associated with the file. The idea is
                    # that we loop over parent directories, and if we find
                    # payloads in the json_data store (indexing by directory
                    # and current file suffix), we check to see if the
                    # candidate JS file's entities are entirely consumed by
                    # the current file. If so, it's a valid candidate, and we
                    # add the payload to the stack. Finally, we invert the
                    # stack and merge the payloads in order.
                    payloads = []
                    target = dirpath
                    while True:
                        if target in json_data and suffix in json_data[target]:
                            for (js_ents, js_md) in json_data[target][suffix]:
                                if not (js_ents - file_ents):
                                    payloads.append(js_md)

                        parent = os.path.dirname(target)
                        if parent == target:
                            break
                        target = parent

                    file_md = {}
                    for pl in payloads[::-1]:
                        file_md.update(pl)

                    # Create database records, including any new Entities
                    for md_key, md_val in file_md.items():
                        if md_key not in all_entities:
                            all_entities[md_key] = Entity(md_key, is_metadata=True)
                            session.add(all_entities[md_key])
                            session.commit()
                        tag = Tag(bf, all_entities[md_key], md_val)
                        session.add(tag)

                session.commit()

            # Recursively index subdirectories
            for d in dirnames:

                d = os.path.join(dirpath, d)

                # Derivative directories must always be added separately and
                # passed as their own root, so terminate if passed.
                if d.startswith(os.path.join(layout.root, 'derivatives')):
                    continue

                # Skip directories that fail validation, unless force_index
                # is defined, in which case we have to keep scanning, in the
                # event that a file somewhere below the current level matches.
                # Unfortunately we probably can't do much better than this
                # without a lot of additional work, because the elements of
                # .force_index can be SRE_Patterns that match files below in
                # unpredictable ways.
                if check_path_matches_patterns(d, layout.force_index):
                    force_index = True
                else:
                    valid_dir = layout._validate_dir(d)
                    # Note the difference between self.force_index and
                    # self.layout.force_index.
                    if not valid_dir and not layout.force_index:
                        continue

                index_dir(d, list(config), path, force_index)

            # prevent subdirectory traversal
            break
    
    index_dir(root_path, config, None, force_index)
