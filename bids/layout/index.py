"""File-indexing functionality. """

import os
import json
from collections import defaultdict
from pathlib import Path
from bids_validator import BIDSValidator
import bids.config
from .models import Config, Entity, Tag, FileAssociation
from ..utils import listify, make_bidsfile
from ..exceptions import BIDSConflictingValuesError


def _extract_entities(bidsfile, entities):
    match_vals = {}
    for e in entities.values():
        m = e.match_file(bidsfile)
        if m is None and e.mandatory:
            break
        if m is not None:
            match_vals[e.name] = (e, m)
    return match_vals


def _check_path_matches_patterns(path, patterns):
    """Check if the path matches at least one of the provided patterns. """
    if not patterns:
        return False
    path = os.path.abspath(path)
    for patt in patterns:
        if isinstance(patt, str):
            if path == patt:
                return True
        elif patt.search(path):
            return True
    return False


class BIDSLayoutIndexer(object):
    """ Indexer class for BIDSLayout.

    Args:
        layout (BIDSLayout): The BIDSLayout to index.
    """

    def __init__(self, layout):

        self.layout = layout
        self.config = list(layout.config.values())
        self.session = layout.session
        self.validate = layout.validate
        self.root = layout.root
        self.config_filename = layout.config_filename
        self.validator = BIDSValidator(index_associated=True)
        # Create copies of list attributes we'll modify during indexing
        self.include_patterns = list(layout.force_index)
        self.exclude_patterns = list(layout.ignore)

    def _validate_dir(self, d, default=None):
        if _check_path_matches_patterns(d, self.include_patterns):
            return True
        if _check_path_matches_patterns(d, self.exclude_patterns):
            return False
        return default

    def _validate_file(self, f, default=None):
        # Inclusion takes priority over exclusion
        if _check_path_matches_patterns(f, self.include_patterns):
            return True
        if _check_path_matches_patterns(f, self.exclude_patterns):
            return False

        # If inclusion/exclusion is inherited from a parent directory, that
        # takes precedence over the remaining file-level rules
        if default is not None:
            return default

        # Derivatives are currently not validated.
        # TODO: raise warning the first time in a session this is encountered
        if not self.validate or 'derivatives' in self.layout.config:
            return True

        # BIDS validator expects absolute paths, but really these are relative
        # to the BIDS project root.
        to_check = os.path.relpath(f, self.root)
        to_check = os.path.join(os.path.sep, to_check)
        to_check = Path(to_check).as_posix()  # bids-validator works with posix paths only
        return self.validator.is_bids(to_check)

    def _index_dir(self, path, config, default_action=None):

        abs_path = os.path.join(self.root, path)

        # Derivative directories must always be added separately
        # and passed as their own root, so terminate if passed.
        if abs_path.startswith(os.path.join(self.root, 'derivatives')):
            return

        config = list(config)  # Shallow copy

        # Check for additional config file in directory
        layout_file = self.config_filename
        config_file = os.path.join(abs_path, layout_file)
        if os.path.exists(config_file):
            cfg = Config.load(config_file, session=self.session)
            config.append(cfg)

        # Track which entities are valid in filenames for this directory
        config_entities = {}
        for c in config:
            config_entities.update(c.entities)

        for (dirpath, dirnames, filenames) in os.walk(path):

            # Set the default inclusion/exclusion directive
            default = self._validate_dir(dirpath, default=default_action)

            # If layout configuration file exists, delete it
            if self.config_filename in filenames:
                filenames.remove(self.config_filename)

            for f in filenames:

                bf = self._index_file(f, dirpath, config_entities,
                                      default_action=default)
                if bf is None:
                    continue

            self.session.commit()

            # Recursively index subdirectories
            for d in dirnames:
                d = os.path.join(dirpath, d)
                self._index_dir(d, list(config), default_action=default)

            # Prevent subdirectory traversal
            break

    def _index_file(self, f, dirpath, entities, default_action=None):
        """Create DB record for file and its tags. """
        abs_fn = os.path.join(dirpath, f)

        # Skip files that fail validation, unless forcibly indexing
        if not self._validate_file(abs_fn, default=default_action):
            return None

        bf = make_bidsfile(abs_fn)
        self.session.add(bf)

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
                self.session.add(tag)

        return bf

    def add_files(self):
        """Index all files in the BIDS dataset. """
        self._index_dir(self.root, self.config)

    def add_metadata(self, **filters):
        """Index metadata for all files in the BIDS dataset.

        Parameters
        ----------

        **filters
            keyword arguments passed to the .get() method of a
            :obj:`bids.layout.BIDSLayout` object.
            These keyword arguments define what files get selected
            for metadata indexing.
        """

        dot = '.' if bids.config.get_option('extension_initial_dot') else ''
        
        if filters:
            # ensure we are returning objects
            filters['return_type'] = 'object'
            # until 0.11.0, user can specify extension or extensions
            ext_key = 'extensions' if 'extensions' in filters else 'extension'
            if filters.get(ext_key):
                filters[ext_key] = listify(filters[ext_key])
                # ensure json files are being indexed
                # XXX 0.14: dot always == '.'
                json_ext = dot + 'json'
                if json_ext not in filters[ext_key]:
                    filters[ext_key].append(json_ext)

        # Process JSON files first if we're indexing metadata
        all_files = self.layout.get(absolute_paths=True, **filters)

        # Track ALL entities we've seen in file names or metadatas
        all_entities = {}
        for c in self.config:
            all_entities.update(c.entities)

        # If key/value pairs in JSON files duplicate ones extracted from files,
        # we can end up with Tag collisions in the DB. To prevent this, we
        # store all filename/entity pairs and the value, and then check against
        # that before adding each new Tag.
        all_tags = {}
        for t in self.session.query(Tag).all():
            key = '{}_{}'.format(t.file_path, t.entity_name)
            all_tags[key] = str(t.value)

        # We build up a store of all file data as we iterate files. It looks
        # like: { extension/suffix: dirname: [(entities, payload)]}}.
        # The payload is left empty for non-JSON files.
        file_data = {}

        for bf in all_files:
            file_ents = bf.entities.copy()
            suffix = file_ents.pop('suffix', None)
            ext = file_ents.pop('extension', None)

            if suffix is not None and ext is not None:
                key = "{}/{}".format(ext, suffix)
                if key not in file_data:
                    file_data[key] = defaultdict(list)

                if ext == dot + 'json':
                    with open(bf.path, 'r') as handle:
                        try:
                            payload = json.load(handle)
                        except json.JSONDecodeError as e:
                            msg = ("Error occurred while trying to decode JSON"
                                   " from file '{}'.".format(bf.path))
                            raise IOError(msg) from e
                else:
                    payload = None

                to_store = (file_ents, payload, bf.path)
                file_data[key][bf.dirname].append(to_store)

        # To avoid integrity errors, track primary keys we've seen
        seen_assocs = set()

        def create_association_pair(src, dst, kind, kind2=None):
            kind2 = kind2 or kind
            pk1 = '#'.join([src, dst, kind])
            if pk1 not in seen_assocs:
                self.session.add(FileAssociation(src=src, dst=dst, kind=kind))
                seen_assocs.add(pk1)
            pk2 = '#'.join([dst, src, kind2])
            if pk2 not in seen_assocs:
                self.session.add(FileAssociation(src=dst, dst=src, kind=kind2))
                seen_assocs.add(pk2)

        # TODO: Efficiency of everything in this loop could be improved
        filenames = [bf for bf in all_files if not bf.path.endswith('.json')]

        for bf in filenames:
            file_ents = bf.entities.copy()
            suffix = file_ents.pop('suffix', None)
            ext = file_ents.pop('extension', None)
            file_ent_keys = set(file_ents.keys())

            if suffix is None or ext is None:
                continue

            # Extract metadata associated with the file. The idea is
            # that we loop over parent directories, and if we find
            # payloads in the file_data store (indexing by directory
            # and current file suffix), we check to see if the
            # candidate JS file's entities are entirely consumed by
            # the current file. If so, it's a valid candidate, and we
            # add the payload to the stack. Finally, we invert the
            # stack and merge the payloads in order.
            ext_key = "{}/{}".format(ext, suffix)
            json_key = dot + "json/{}".format(suffix)
            dirname = bf.dirname

            payloads = []
            ancestors = []

            while True:
                # Get JSON payloads
                json_data = file_data.get(json_key, {}).get(dirname, [])
                for js_ents, js_md, js_path in json_data:
                    js_keys = set(js_ents.keys())
                    if js_keys - file_ent_keys:
                        continue
                    matches = [js_ents[name] == file_ents[name]
                               for name in js_keys]
                    if all(matches):
                        payloads.append((js_md, js_path))

                # Get all files this file inherits from
                candidates = file_data.get(ext_key, {}).get(dirname, [])
                for ents, _, path in candidates:
                    keys = set(ents.keys())
                    if keys - file_ent_keys:
                        continue
                    matches = [ents[name] == file_ents[name] for name in keys]
                    if all(matches):
                        ancestors.append(path)

                parent = os.path.dirname(dirname)
                if parent == dirname:
                    break
                dirname = parent

            if not payloads:
                continue

            # Create DB records for metadata associations
            js_file = payloads[-1][1]
            create_association_pair(js_file, bf.path, 'Metadata')

            # Consolidate metadata by looping over inherited JSON files
            file_md = {}
            for pl, js_file in payloads[::-1]:
                file_md.update(pl)

            # Create FileAssociation records for JSON inheritance
            n_pl = len(payloads)
            for i, (pl, js_file) in enumerate(payloads):
                if (i + 1) < n_pl:
                    other = payloads[i + 1][1]
                    create_association_pair(js_file, other, 'Child', 'Parent')

            # Inheritance for current file
            n_pl = len(ancestors)
            for i, src in enumerate(ancestors):
                if (i + 1) < n_pl:
                    dst = ancestors[i + 1]
                    create_association_pair(src, dst, 'Child', 'Parent')

            # Files with IntendedFor field always get mapped to targets
            intended = listify(file_md.get('IntendedFor', []))
            for target in intended:
                # Per spec, IntendedFor paths are relative to sub dir.
                target = os.path.join(
                    self.root, 'sub-{}'.format(bf.entities['subject']), target)
                create_association_pair(bf.path, target, 'IntendedFor',
                                        'InformedBy')

            # Link files to BOLD runs
            if suffix in ['physio', 'stim', 'events', 'sbref']:
                images = self.layout.get(
                    extension=['.nii', '.nii.gz'], suffix='bold',
                    return_type='filename', **file_ents)
                for img in images:
                    create_association_pair(bf.path, img, 'IntendedFor',
                                            'InformedBy')

            # Link files to DWI runs
            if suffix == 'sbref' or ext in ['bvec', 'bval']:
                images = self.layout.get(
                    extension=['.nii', '.nii.gz'], suffix='dwi',
                    return_type='filename', **file_ents)
                for img in images:
                    create_association_pair(bf.path, img, 'IntendedFor',
                                            'InformedBy')

            # Create Tag <-> Entity mappings, and any newly discovered Entities
            for md_key, md_val in file_md.items():
                tag_string = '{}_{}'.format(bf.path, md_key)
                # Skip pairs that were already found in the filenames
                if tag_string in all_tags:
                    file_val = all_tags[tag_string]
                    if str(md_val) != file_val:
                        msg = (
                            "Conflicting values found for entity '{}' in "
                            "filename {} (value='{}') versus its JSON sidecar "
                            "(value='{}'). Please reconcile this discrepancy."
                        )
                        raise BIDSConflictingValuesError(
                            msg.format(md_key, bf.path, file_val,
                            md_val))
                    continue
                if md_key not in all_entities:
                    all_entities[md_key] = Entity(md_key, is_metadata=True)
                    self.session.add(all_entities[md_key])
                tag = Tag(bf, all_entities[md_key], md_val)
                self.session.add(tag)

            if len(self.session.new) >= 1000:
                self.session.commit()

        self.session.commit()
