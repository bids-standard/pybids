"""Utility functions."""

import re
import os
import json
import requests
from tempfile import NamedTemporaryFile
from pathlib import Path
from frozendict import frozendict as _frozendict
from upath import UPath as Path
from functools import cache


# Monkeypatch to print out frozendicts *as if* they were dictionaries.
class frozendict(_frozendict):
    """A hashable dictionary type."""

    def __repr__(self):
        """Override frozendict representation."""
        return repr({k: v for k, v in self.items()})


def listify(obj):
    """Wraps all non-list or tuple objects in a list; provides a simple way
    to accept flexible arguments."""
    return obj if isinstance(obj, (list, tuple, type(None))) else [obj]


def hashablefy(obj):
    """Make dictionaries and lists hashable or raise."""
    if isinstance(obj, list):
        return tuple([hashablefy(o) for o in obj])

    if isinstance(obj, dict):
        return frozendict({k: hashablefy(v) for k, v in obj.items()})
    return obj


def matches_entities(obj, entities, strict=False):
    """Checks whether an object's entities match the input."""
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
    """
    based on snippet found at https://stackoverflow.com/a/4836734/2445984
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        if field is not None:
            key = getattr(key, field)
        if not isinstance(key, str):
            key = str(key)
        return [convert(c) for c in re.split("([0-9]+)", key)]

    return sorted(l, key=alphanum_key)


def convert_JSON(j):
    """Recursively convert CamelCase keys to snake_case.
    From: https://stackoverflow.com/questions/17156078/
    converting-identifier-naming-between-camelcase-and-
    underscores-during-json-seria
    """

    def camel_to_snake(s):
        a = re.compile("((?<=[a-z0-9])[A-Z]|(?!^)[A-Z](?=[a-z]))")
        return a.sub(r"_\1", s).lower()

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
        if isinstance(value, dict) and k != "Replace":
            out[newK] = convert_JSON(value)
        elif isinstance(value, list):
            out[newK] = convertArray(value)
        else:
            out[newK] = value

    return out


def splitext(path):
    """splitext for paths with directories that may contain dots.
    From https://stackoverflow.com/questions/5930036/separating-file-extensions-using-python-os-path-module
    """
    li = []
    path_without_extensions = os.path.join(
        os.path.dirname(path), os.path.basename(path).split(os.extsep)[0]
    )
    extensions = os.path.basename(path).split(os.extsep)[1:]
    li.append(path_without_extensions)
    # li.append(extensions) if you want extensions in another list inside the list that is returned.
    li.extend(extensions)
    return li


def make_bidsfile(filename):
    """Create a BIDSFile instance of the appropriate class."""
    from .layout import models

    # Extract all extensions from filename (a.tar.gz -> .tar.gz, not just .gz)
    ext = "".join(Path(filename).suffixes)

    if ext.endswith((".nii", ".nii.gz", ".gii")):
        cls = "BIDSImageFile"
    elif ext in [".tsv", ".tsv.gz"]:
        cls = "BIDSDataFile"
    elif ext == ".json":
        cls = "BIDSJSONFile"
    else:
        cls = "BIDSFile"

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
    MULTICONTRAST_ENTITIES = ["echo", "part", "ch", "direction"]
    MULTICONTRAST_SUFFIXES = [
        ("bold", "phase"),
        ("phase1", "phase2", "phasediff", "magnitude1", "magnitude2"),
    ]
    if len(extra_entities):
        MULTICONTRAST_ENTITIES += extra_entities

    collected_files = []
    for f in files:
        if len(collected_files) and any(
            f in filegroup for filegroup in collected_files
        ):
            continue
        ents = f.get_entities()
        ents = {k: v for k, v in ents.items() if k not in MULTICONTRAST_ENTITIES}

        # Group files with differing multi-contrast entity values, but same
        # everything else.
        all_suffixes = ents["suffix"]
        for mcs in MULTICONTRAST_SUFFIXES:
            if ents["suffix"] in mcs:
                all_suffixes = mcs
                break
        ents.pop("suffix")
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


@cache
def entity_indices(schema_spec=None):
    from bidsschematools.schema import load_schema
    from collections import defaultdict

    entities = load_schema(schema_spec).rules.entities + [
        "suffix",
        "extension",
        "datatype",
    ]

    return defaultdict(
        lambda e=entities: len(e), {elem: idx for idx, elem in enumerate(entities)}
    )


def bids_sort(unsorted: dict, schema_spec=None):
    f"""
    Sorts filename entity dictionaries according to their order as defined in 
    schema.rules.entities as well as suffix, extension. Lastly, appends datatype
    to the end of the sort to accommodate pybids datastructures.

    Parameters
    ----------
    unsorted: dict
        A dictionary containing bids file entities and their values.
    schema_spec: str
        Path or version of schema to use, defaults to the version bundled
        with bidsschematools.
    
    Returns
    -------
    sorted_bids: dict

    """
    indices = entity_indices(schema_spec)

    return {k: unsorted[k] for k in sorted(unsorted, key=indices.__getitem__)}


def _allowed_bids_versions(timeout=5, min_version="1.8.0"):
    """Fetch BIDS specification releases from GitHub, strip leading 'v', filter to >= min_version.
    Returns a set of version strings (e.g. {'1.9.0', '1.10.0', ...}) or None on timeout/error.
    """
    try:
        from packaging.version import Version

        r = requests.get(
            "https://api.github.com/repos/bids-standard/bids-specification/releases",
            params={"per_page": 100},
            timeout=timeout,
        )
        if r.status_code != 200:
            return None
        min_ver = Version(min_version)
        allowed = set()
        for release in r.json():
            tag = release.get("tag_name", "")
            ver_str = tag.lstrip("v")
            try:
                if Version(ver_str) >= min_ver:
                    allowed.add(ver_str)
            except Exception:
                continue
        return allowed if allowed else None
    except (requests.RequestException, ValueError):
        return None


def collect_schema(
    uri: str = None,
    bids_version: str = None,
    schema_version: str = None,
):
    if uri is not None and bids_version is not None:
        raise ValueError(
            "uri and bids_version are mutually exclusive, "
            f"you gave uri={uri!r}, bids_version={bids_version!r}"
        )

    if bids_version and not uri:
        version_pattern = re.compile(r"(?<![.\d])\d+(?:\.\d+){1,}(?![.\d])")
        version = version_pattern.search(bids_version)
        if version:
            version = f"v{version.group(0)}"
        if 'latest' in bids_version:
            version = 'latest'
        if 'stable' in bids_version:
            version = 'stable'
        if not version:
            raise ValueError(f"Unable to determine version from bids_version={bids_version}")

        # Validate numeric version against available releases (>= 1.8.0); fail gracefully on timeout
        if version not in ("latest", "stable"):
            allowed = _allowed_bids_versions()
            if allowed is not None and version.lstrip("v") not in allowed:
                raise ValueError(
                    f"bids_version {bids_version!r} (resolved to {version!r}) is not an available "
                    f"BIDS release >= 1.8.0. Available: {', '.join(sorted(allowed))}"
                )

        uri = f"https://bids-specification.readthedocs.io/en/{version}/schema.json"
    if uri is None:
        uri = "https://bids-specification.readthedocs.io/en/latest/schema.json"

    from bidsschematools.schema import load_schema

    schema = load_schema(Path(uri))

    return schema


# Alias for use by layout.models.Config._from_schema
get_schema = collect_schema
