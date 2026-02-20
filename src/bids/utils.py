""" Utility functions. """

import re
import os
from pathlib import Path
from frozendict import frozendict as _frozendict
from sqlalchemy import schema
from upath import UPath as Path
from typing import Literal


# we'll be reusing this
bids_schema = None

def get_schema(path: Literal["latest", "stable", "bundled"] | Path | None = "stable", fail_silently=False):
    """Load the BIDS schema.

    By default loads the schema from the BIDS specification "stable" docs.
    Use ``"latest"`` or ``"stable"`` to fetch from the specification website,
    or pass a path (local or URI) to load a specific schema. Schema is cached;
    check ``get_schema().schema_version`` and/or ``get_schema().bids_version``
    to confirm the loaded version.

    Parameters
    ----------
    path : {"latest", "stable", "bundled"} or Path or None, optional
        Source for the schema. Use ``"latest"`` or ``"stable"`` to fetch from
        the specification website, or ``"bundled"`` / None to use the schema
        packaged with bidsschematools. A path or URI to a schema file or
        directory may also be passed. Default is ``"stable"``.
    fail_silently : bool, optional
        If True, on failure to retrieve the requested schema (e.g. no network),
        fall back to the schema packaged with bidsschematools instead of
        raising. Default is False.

    Returns
    -------
    dict-like
        The BIDS schema (e.g. with ``objects``, ``rules``, ``schema_version``,
        ``bids_version``).
    """
    global bids_schema
    if bids_schema is not None:
        return bids_schema

    from bidsschematools.schema import load_schema
    from bidsschematools.types.namespace import Namespace
    import requests

    _url = "https://bids-specification.readthedocs.io/en/{version}/schema.json"

    # Resolve what to try: URL for "latest"/"stable", path as-is, or None for bundled only
    if path in ("latest", "stable"):
        source = _url.format(version=path)
    elif path is None or path == "bundled":
        source = None
    else:
        source = path  # Path or path-like

    if source is None:
        bids_schema = load_schema()
        return bids_schema

    # Fetch URLs with requests. UPath/fsspec uses aiohttp which can give
    # FileNotFoundError for these URLs; requests with default cert verification works.
    # SSL verification: config key schema_verify_ssl (pybids_config.json) or env
    # BIDS_SCHEMA_VERIFY_SSL=0 to disable (e.g. behind a proxy).
    if isinstance(source, str) and source.startswith("http"):
        try:
            if "BIDS_SCHEMA_VERIFY_SSL" in os.environ:
                verify = os.environ.get("BIDS_SCHEMA_VERIFY_SSL", "1").lower() not in ("0", "false", "no")
            else:
                from . import config as _bids_config
                verify = _bids_config.get_option("schema_verify_ssl")
            resp = requests.get(source, timeout=30, verify=verify)
            resp.raise_for_status()
            bids_schema = Namespace.from_json(resp.text)
        except Exception as err:
            if fail_silently:
                bids_schema = load_schema()
            else:
                raise err
    else:
        try:
            bids_schema = load_schema(source)
        except Exception as err:
            if fail_silently:
                bids_schema = load_schema()
            else:
                raise err
    return bids_schema

# Monkeypatch to print out frozendicts *as if* they were dictionaries.
class frozendict(_frozendict):
    """A hashable dictionary type."""

    def __repr__(self):
        """Override frozendict representation."""
        return repr({k: v for k, v in self.items()})


def listify(obj):
    ''' Wraps all non-list or tuple objects in a list; provides a simple way
    to accept flexible arguments. '''
    return obj if isinstance(obj, (list, tuple, type(None))) else [obj]


def hashablefy(obj):
    ''' Make dictionaries and lists hashable or raise. '''
    if isinstance(obj, list):
        return tuple([hashablefy(o) for o in obj])

    if isinstance(obj, dict):
        return frozendict({k: hashablefy(v) for k, v in obj.items()})
    return obj


def matches_entities(obj, entities, strict=False):
    ''' Checks whether an object's entities match the input. '''
    if strict and set(obj.entities.keys()) != set(entities.keys()):
        return False

    comm_ents = list(set(obj.entities.keys()) & set(entities.keys()))
    for k in comm_ents:
        current = obj.entities[k]
        target = entities[k]
        if isinstance(target, (list, tuple)):
            if current not in target:
                return False
        elif current != target:
            return False
    return True


def natural_sort(l, field=None):
    '''
    based on snippet found at https://stackoverflow.com/a/4836734/2445984
    '''
    convert = lambda text: int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        if field is not None:
            key = getattr(key, field)
        if not isinstance(key, str):
            key = str(key)
        return [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def convert_JSON(j):
    """ Recursively convert CamelCase keys to snake_case.
    From: https://stackoverflow.com/questions/17156078/
    converting-identifier-naming-between-camelcase-and-
    underscores-during-json-seria
    """
    def camel_to_snake(s):
        a = re.compile('((?<=[a-z0-9])[A-Z]|(?!^)[A-Z](?=[a-z]))')
        return a.sub(r'_\1', s).lower()

    def convertArray(a):
        newArr = []
        for i in a:
            if isinstance(i, list):
                newArr.append(convertArray(i))
            elif isinstance(i, dict):
                newArr.append(convert_JSON(i))
            else:
                newArr.append(i)
        return newArr

    out = {}
    for k, value in j.items():
        newK = camel_to_snake(k)

        # Replace transformation uses a dict, so skip lower-casing
        if isinstance(value, dict) and k != 'Replace':
            out[newK] = convert_JSON(value)
        elif isinstance(value, list):
            out[newK] = convertArray(value)
        else:
            out[newK] = value

    return out


def splitext(path):
    """splitext for paths with directories that may contain dots.
    From https://stackoverflow.com/questions/5930036/separating-file-extensions-using-python-os-path-module"""
    li = []
    path_without_extensions = os.path.join(os.path.dirname(path),
        os.path.basename(path).split(os.extsep)[0])
    extensions = os.path.basename(path).split(os.extsep)[1:]
    li.append(path_without_extensions)
    # li.append(extensions) if you want extensions in another list inside the list that is returned.
    li.extend(extensions)
    return li


def make_bidsfile(filename):
    """Create a BIDSFile instance of the appropriate class. """
    from .layout import models

    # Extract all extensions from filename (a.tar.gz -> .tar.gz, not just .gz)
    ext = ''.join(Path(filename).suffixes)

    if ext.endswith(('.nii', '.nii.gz', '.gii')):
        cls = 'BIDSImageFile'
    elif ext in ['.tsv', '.tsv.gz']:
        cls = 'BIDSDataFile'
    elif ext == '.json':
        cls = 'BIDSJSONFile'
    else:
        cls = 'BIDSFile'

    Cls = getattr(models, cls)
    return Cls(filename)


def collect_associated_files(layout, files, extra_entities=()):
    """Collect and group BIDSFiles with multiple files per acquisition.

    Parameters
    ----------
    layout
    files : list of BIDSFile
    extra_entities

    Returns
    -------
    collected_files : list of list of BIDSFile
    """
    MULTICONTRAST_ENTITIES = ['echo', 'part', 'ch', 'direction']
    MULTICONTRAST_SUFFIXES = [
        ('bold', 'phase'),
        ('phase1', 'phase2', 'phasediff', 'magnitude1', 'magnitude2'),
    ]
    if len(extra_entities):
        MULTICONTRAST_ENTITIES += extra_entities

    collected_files = []
    for f in files:
        if len(collected_files) and any(f in filegroup for filegroup in collected_files):
            continue
        ents = f.get_entities()
        ents = {k: v for k, v in ents.items() if k not in MULTICONTRAST_ENTITIES}

        # Group files with differing multi-contrast entity values, but same
        # everything else.
        all_suffixes = ents['suffix']
        for mcs in MULTICONTRAST_SUFFIXES:
            if ents['suffix'] in mcs:
                all_suffixes = mcs
                break
        ents.pop('suffix')
        associated_files = layout.get(suffix=all_suffixes, **ents)
        collected_files.append(associated_files)
    return collected_files


def validate_multiple(val, retval=None):
    """Any click.Option with the multiple flag will return an empty tuple if not set.

    This helper method converts empty tuples to a desired return value (default: None).
    This helper method selects the first item in single-item tuples.
    """
    assert isinstance(val, tuple)

    if val == tuple():
        return retval
    if len(val) == 1:
        return val[0]
    return val

def bids_sort(unsorted: dict):
    _schema = get_schema()
    entity_order = list(_schema.rules.entities) + ['suffix', 'extension', 'datatype']
    
    sorted_bids = {k: unsorted[k] for k in sorted(unsorted, key=lambda k: entity_order.index(k) if k in entity_order else len(entity_order))}

    return sorted_bids
