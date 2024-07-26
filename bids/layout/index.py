"""File-indexing functionality. """

import os
import json
import re
import fsspec
from collections import defaultdict
from upath import UPath as Path
from functools import partial, lru_cache

from bids_validator import BIDSValidator

from ..utils import listify, make_bidsfile
from ..exceptions import BIDSConflictingValuesError

from .models import Config, Entity, Tag, FileAssociation, _create_tag_dict
from .validation import validate_indexing_args

def _regexfy(patt, root=None):
    if hasattr(patt, "search"):
        return patt

    patt = Path(patt)

    if patt.is_absolute():
        patt = str(patt.relative_to(root or "/"))

    return re.compile(r"^/" + str(patt) + r".*")


def _extract_entities(bidsfile, entities):
    match_vals = {}
    for e in entities.values():
        m = e.match_file(bidsfile)
        if m is None and e.mandatory:
            break
        if m is not None:
            match_vals[e.name] = (e, m)
    return match_vals


def _check_path_matches_patterns(path, patterns, root=None):
    """Check if the path matches at least one of the provided patterns. """
    if not patterns:
        return False

    path = path.absolute()
    if root is not None:
        path = Path("/") / path.relative_to(root)

    # Path now can be downcast to str
    path = str(path)

    for patt in patterns:
        if patt.search(path):
            return True
    return False


def _validate_path(path, incl_patt=None, excl_patt=None, root=None):
    if _check_path_matches_patterns(path, incl_patt, root=root):
        return True

    if _check_path_matches_patterns(path, excl_patt, root=root):
        return False


class BIDSLayoutIndexer:
    """ Indexer class for BIDSLayout.

    Parameters
    ----------
    validate : bool, optional
        If True, all files are checked for BIDS compliance when first indexed,
        and non-compliant files are ignored. This provides a convenient way to
        restrict file indexing to only those files defined in the "core" BIDS
        spec, as setting ``validate=True`` will lead noncompliant files
        like ``sub-01/nonbidsfile.txt`` to be ignored.
    ignore : str or SRE_Pattern or list
        Path(s) to exclude from indexing. Each path is either a string or a
        SRE_Pattern object (i.e., compiled regular expression). If a string is
        passed, it must be either an absolute path, or be relative to the BIDS
        project root. If an SRE_Pattern is passed, the contained regular
        expression will be matched against the full (absolute) path of all
        files and directories. By default, indexing ignores all files in
        'code/', 'stimuli/', 'sourcedata/', 'models/', and any hidden
        files/dirs beginning with '.' at root level.
    force_index : str or SRE_Pattern or list
        Path(s) to forcibly index in the BIDSLayout, even if they would
        otherwise fail validation. See the documentation for the ignore
        argument for input format details. Note that paths in force_index takes
        precedence over those in ignore (i.e., if a file matches both ignore
        and force_index, it *will* be indexed).
        Note: NEVER include 'derivatives' here; use the derivatives argument
        (or :obj:`bids.layout.BIDSLayout.add_derivatives`) for that.
    index_metadata : bool
        If True, all metadata files are indexed. If False, metadata will not be
        available (but indexing will be faster).
    config_filename : str
        Optional name of filename within directories
        that contains configuration information.
    **filters
        keyword arguments passed to the .get() method of a
        :obj:`bids.layout.BIDSLayout` object. These keyword arguments define
        what files get selected for metadata indexing.
    """

    def __init__(
        self,
        validate=False,
        ignore=None,
        force_index=None,
        index_metadata=True,
        config_filename='layout_config.json',
        **filters,
    ):
        self.ignore = ignore
        self.force_index = force_index
        self.index_metadata = index_metadata
        self.config_filename = config_filename
        self.filters = filters
        self.validator = None

        if validate:
            self.validator = BIDSValidator(index_associated=True)

        # Layout-dependent attributes to be set in __call__()
        self._layout = None
        self._config = None
        self._include_patterns = None
        self._exclude_patterns = None

    def __call__(self, layout):
        self._layout = layout
        self._config = list(layout.config.values())

        ignore, force = validate_indexing_args(self.ignore, self.force_index,
                                               self._layout._root)

        # Do not accept string patterns
        self._include_patterns = [
            _regexfy(patt, root=self._layout._root) for patt in listify(force)
        ]
        self._exclude_patterns = [
            _regexfy(patt, root=self._layout._root) for patt in listify(ignore)
        ]

        all_bfs, all_tag_dicts = self._index_dir(self._layout._root, self._config)

        self.session.bulk_save_objects(all_bfs)
        self.session.bulk_insert_mappings(Tag, all_tag_dicts)
        self.session.commit()

        if self.index_metadata:
            self._index_metadata()

    @property
    def session(self):
        return self._layout.connection_manager.session

    def _validate_file(self, f):
        matched_patt = _validate_path(
            f,
            incl_patt=self._include_patterns,
            excl_patt=self._exclude_patterns,
            root=self._layout._root
        )

        if matched_patt is not None:
            return matched_patt

        if self.validator is None:
            return True

        # BIDS validator expects absolute paths, but really these are relative
        # to the BIDS project root.
        to_check = Path(f.path).relative_to(Path(self._layout._root.path)) # use .path then Path() to drop the uri prefix
        # Pretend the path is an absolute path
        to_check = Path('/') / to_check
        # bids-validator works with posix paths only
        to_check = to_check.as_posix()
        return self.validator.is_bids(to_check)

    def _index_dir(self, path, config, force=None):

        abs_path = self._layout._root / path

        # Derivative directories must always be added separately
        if self._layout._root.joinpath('derivatives') in abs_path.parents:
            return [], []

        config = list(config)  # Shallow copy

        # Check for additional config file in directory
        layout_file = self.config_filename
        config_file = abs_path / layout_file
        if config_file.exists():
            cfg = Config.load(config_file, session=self.session)
            config.append(cfg)

        # Track which entities are valid in filenames for this directory
        config_entities = {}
        for c in config:
            config_entities.update(c.entities)


        # Get lists of 1st-level subdirectories and files in the path directory
        _, dirnames, filenames = next(path.fs.walk(path.path))

        # Move any directories suffixed with .zarr to filenames
        zarrnames = [entry for entry in dirnames if Path(entry).suffix == '.zarr']
        dirnames = [entry for entry in dirnames if Path(entry).suffix != '.zarr']
        filenames.extend(zarrnames)

        # If layout configuration file exists, delete it
        if self.config_filename in filenames:
            filenames.remove(self.config_filename)

        all_bfs = []
        all_tag_dicts = []
        for f in filenames:
            abs_fn = path / f
            # Skip files that fail validation, unless forcibly indexing
            if force or self._validate_file(abs_fn):
                bf, tag_dicts = self._index_file(abs_fn, config_entities)
                all_tag_dicts += tag_dicts
                all_bfs.append(bf)

        # Recursively index subdirectories
        for d in dirnames:
            d = path / d
            force = _validate_path(
                d,
                incl_patt=self._include_patterns,
                excl_patt=self._exclude_patterns,
                root=self._layout._root,
            )
            if force is not False:
                dir_bfs, dir_tag_dicts = self._index_dir(d, config, force=force)
                all_bfs += dir_bfs
                all_tag_dicts += dir_tag_dicts

        return all_bfs, all_tag_dicts

    def _index_file(self, abs_fn, entities):
        """Create DB record for file and its tags. """
        bf = make_bidsfile(abs_fn)

        # Extract entity values
        match_vals = {}
        for e in entities.values():
            m = e.match_file(bf)
            if m is None and e.mandatory:
                break
            if m is not None:
                match_vals[e.name] = (e, m)

        # Create Entity <=> BIDSFile mappings
        tag_dicts = [
            _create_tag_dict(bf, ent, val, ent._dtype)
            for ent, val in match_vals.values()
        ]

        return bf, tag_dicts

    def _index_metadata(self):
        """Index metadata for all files in the BIDS dataset.
        """
        filters = self.filters

        if filters:
            # ensure we are returning objects
            filters['return_type'] = 'object'

            if filters.get('extension'):
                filters['extension'] = listify(filters['extension'])
                # ensure json files are being indexed
                if '.json' not in filters['extension']:
                    filters['extension'].append('.json')

        # Process JSON files first if we're indexing metadata
        all_files = self._layout.get(absolute_paths=True, **filters)

        # Track ALL entities we've seen in file names or metadatas
        all_entities = {}
        for c in self._config:
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

        # Memoizing JSON loader
        # Use as a function to allow lazy loading so only read JSON files
        # if they correspond to data files that are indexed
        @lru_cache(maxsize=None)
        def load_json(path):
            with Path(path).fs.open(Path(path).path, 'r', encoding='utf-8') as handle:
                try:
                    return json.load(handle)
                except (UnicodeDecodeError, json.JSONDecodeError) as e:
                    raise OSError(
                        "Error occurred while trying to decode JSON "
                        f"from file {path}"
                    ) from e

        filenames = []
        for bf in all_files:
            if 'suffix' in bf.entities and 'extension' in bf.entities:
                file_ents = bf.entities.copy()
                suffix = file_ents.pop('suffix')
                ext = file_ents.pop('extension')
                key = "{}/{}".format(ext, suffix)
                if key not in file_data:
                    file_data[key] = defaultdict(list)

                payload = None
                if ext == '.json':
                    payload = partial(load_json, bf)
                else:
                    filenames.append(bf)

                to_store = (file_ents, payload, bf.path)
                file_data[key][bf._dirname].append(to_store)
                
        # To avoid integrity errors, track primary keys we've seen
        seen_assocs = set()

        def create_association_pair(src, dst, kind, kind2=None):
            objs = []
            kind2 = kind2 or kind
            pk1 = '#'.join([src, dst, kind])
            if pk1 not in seen_assocs:
                objs.append(FileAssociation(src=src, dst=dst, kind=kind))
                seen_assocs.add(pk1)
            pk2 = '#'.join([dst, src, kind2])
            if pk2 not in seen_assocs:
                objs.append((FileAssociation(src=dst, dst=src, kind=kind2)))
                seen_assocs.add(pk2)

            return objs

        all_objs = []
        all_tag_dicts = []
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
            json_key = ".json/{}".format(suffix)
            dirname = bf._dirname

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

                parent = dirname.parent

                if parent == dirname:
                    break
                dirname = parent

            if not payloads:
                continue

            # Missing data files can tolerate absent metadata files,
            # but we will try to load it anyway
            virtual_datafile = not bf._path.exists()

            # Create DB records for metadata associations
            js_file = payloads[0][1]
            all_objs += create_association_pair(js_file, bf.path, 'Metadata')

            # Consolidate metadata by looping over inherited JSON files
            file_md = {}
            for pl, js_file in payloads[::-1]:
                try:
                    file_md.update(pl())
                except FileNotFoundError:
                    if not virtual_datafile:
                        raise
                    # Drop metadata if any files are missing
                    # Otherwise missing overrides could give misleading metadata
                    file_md = {}
                    break

            # Create FileAssociation records for JSON inheritance
            n_pl = len(payloads)
            for i, (pl, js_file) in enumerate(payloads):
                if (i + 1) < n_pl:
                    other = payloads[i + 1][1]
                    all_objs += create_association_pair(js_file, other, 'Child', 'Parent')

            # Inheritance for current file
            n_pl = len(ancestors)
            for i, src in enumerate(ancestors):
                if (i + 1) < n_pl:
                    dst = ancestors[i + 1]
                    all_objs += create_association_pair(src, dst, 'Child', 'Parent')

            # Files with IntendedFor field always get mapped to targets
            intended = listify(file_md.get('IntendedFor', []))
            for target in intended:
                # Per spec, IntendedFor paths are relative to sub dir.
                target = self._layout._root.joinpath(
                    'sub-{}'.format(bf.entities['subject']),
                    target)
                all_objs += create_association_pair(bf.path, str(target), 'IntendedFor',
                                        'InformedBy')

            # Link files to BOLD runs
            if suffix in ['physio', 'stim', 'events', 'sbref']:
                images = self._layout.get(
                    extension=['.nii', '.nii.gz'], suffix='bold',
                    return_type='filename', **file_ents)
                for img in images:
                    all_objs += create_association_pair(bf.path, img, 'IntendedFor',
                                            'InformedBy')

            # Link files to DWI runs
            if suffix == 'sbref' or ext in ['bvec', 'bval']:
                images = self._layout.get(
                    extension=['.nii', '.nii.gz'], suffix='dwi',
                    return_type='filename', **file_ents)
                for img in images:
                    all_objs += create_association_pair(bf.path, img, 'IntendedFor',
                                            'InformedBy')

            # Create Tag <-> Entity mappings, and any newly discovered Entities
            for md_key, md_val in file_md.items():
                # Treat null entries (deserialized to None) as absent
                # Alternative is to cast None to null in layout.models._create_tag_dict
                if md_val is None:
                    continue
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
                            msg.format(md_key, bf.path, file_val, md_val))
                    continue
                if md_key not in all_entities:
                    all_entities[md_key] = Entity(md_key)
                    self.session.add(all_entities[md_key])
                tag = _create_tag_dict(bf, all_entities[md_key], md_val, is_metadata=True)
                all_tag_dicts.append(tag)
                
        self.session.bulk_save_objects(all_objs)
        self.session.bulk_insert_mappings(Tag, all_tag_dicts)
        self.session.commit()

