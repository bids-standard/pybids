""" Utility functions. """

import re
import os
import json
from pathlib import Path
from packaging.version import Version


def listify(obj):
    """ Wraps all non-list or tuple objects in a list; provides a simple way
    to accept flexible arguments. """
    return obj if isinstance(obj, (list, tuple, type(None))) else [obj]


def matches_entities(obj, entities, strict=False):
    """ Checks whether an object's entities match the input. """
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
     based on snippet found at http://stackoverflow.com/a/4836734/2445984
    """
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
    From: https://stackoverflow.com/questions/17156078/converting-identifier-
          naming-between-camelcase-and-underscores-during-json-seria
    """

    def camel_to_snake(s):
        a = re.compile('((?<=[a-z0-9])[A-Z]|(?!^)[A-Z](?=[a-z]))')
        return a.sub(r'_\1', s).lower()

    def convertArray(a):
        newArr = []
        for i in a:
            if isinstance(i,list):
                newArr.append(convertArray(i))
            elif isinstance(i, dict):
                newArr.append(convert_JSON(i))
            else:
                newArr.append(i)
        return newArr

    out = {}
    for k, value in j.items():
        newK = camel_to_snake(k)

        if isinstance(value, dict):
            out[newK] = convert_JSON(value)
        elif isinstance(value, list):
            out[newK] = convertArray(value)
        else:
            out[newK] = value

    return out


def splitext(path):
    """splitext for paths with directories that may contain dots.
    From https://stackoverflow.com/questions/5930036/separating-file
         -extensions-using-python-os-path-module"""
    li = []
    path_without_extensions = os.path.join(os.path.dirname(path),
                                           os.path.basename(path).split(os.extsep)[0])
    extensions = os.path.basename(path).split(os.extsep)[1:]
    li.append(path_without_extensions)
    # li.append(extensions) if you want extensions in another
    # list inside the list that is returned.
    li.extend(extensions)
    return li


def make_bidsfile(filename):
    """Create a BIDSFile instance of the appropriate class. """
    from .layout import models

    patt = re.compile("[._]*[a-zA-Z0-9]*?\\.([^/\\\\]+)$")
    m = re.search(patt, filename)

    ext = None if not m else m.group(1)

    if ext in ['nii', 'nii.gz']:
        cls = 'BIDSImageFile'
    elif ext in ['tsv', 'tsv.gz']:
        cls = 'BIDSDataFile'
    elif ext == 'json':
        cls = 'BIDSJSONFile'
    else:
        cls = 'BIDSFile'

    Cls = getattr(models, cls)
    return Cls(filename)


# As per https://bids.neuroimaging.io/bids_spec1.1.1.pdf
desc_fields = {
    Version("1.1.1"): {
        "required": ["Name", "BIDSVersion"],
        "recommended": ["License"],
        "optional": ["Authors", "Acknowledgements", "HowToAcknowledge",
                     "Funding", "ReferencesAndLinks", "DatasetDOI"]
    }
}


def get_description_fields(version, type_):
    if isinstance(version, str):
        version = Version(version)
    if not isinstance(version, Version):
        raise TypeError("Version must be a string or a packaging.version.Version object.")

    if version in desc_fields:
        return desc_fields[version][type_]
    return desc_fields[max(desc_fields.keys())][type_]


def write_derivative_description(source_dir, name, bids_version='1.1.1', exist_ok=False, **desc_kwargs):

    """
     Write a dataset_description.json file for a new derivative folder.
       source_dir : Directory of the BIDS dataset that has been derived.
                    This dataset can itself be a derivative.
       name : Name of the derivative dataset.
       bid_version: Version of the BIDS standard.
       desc_kwargs: Dictionary of entries that should be added to the
                    dataset_description.json file.
       exist_ok : Control the behavior of pathlib.Path.mkdir when a derivative folder
                  with this name already exists.
    """
    if source_dir is str:
        source_dir = Path(source_dir)

    deriv_dir = source_dir / "derivatives" / name

    # I found nothing about the requirement of a PipelineDescription.Name
    # for derivatives in https://bids.neuroimaging.io/bids_spec1.1.1.pdf, but it
    # is required by BIDSLayout(..., derivatives=True)
    desc = {
        'Name': name,
        'BIDSVersion': bids_version,
        'PipelineDescription': {
            "Name": name
            }
        }
    desc.update(desc_kwargs)

    fname = source_dir / 'dataset_description.json'
    if not fname.exists():
        raise ValueError("The argument source_dir must point to a valid BIDS directory." +
                         "As such, it should contain a dataset_description.json file.")
    with fname.open() as fobj:
        orig_desc = json.load(fobj)

    for field_type in ["recommended", "optional"]:
        for field in get_description_fields(bids_version, field_type):
            if field in desc:
                continue
            if field in orig_desc:
                desc[field] = orig_desc[field]

    for field in get_description_fields(bids_version, "required"):
        if field not in desc:
            raise ValueError("The field {} is required and is currently missing.".format(field))

    deriv_dir.mkdir(parents=True, exist_ok=exist_ok)
    with (deriv_dir / 'dataset_description.json').open('w') as fobj:
        json.dump(desc, fobj, indent=4)
