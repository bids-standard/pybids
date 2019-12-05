import json
from pathlib import Path
from packaging.version import Version


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


def write_derivative_description(source_dir, name, bids_version='1.1.1', exist_ok=False,
                                 propagate=False, **desc_kwargs):
    """Write a dataset_description.json file for a new derivative folder.

    Parameters
    ----------
    source_dir : str or Path
        Directory of the BIDS dataset that has been derived.
        This dataset can itself be a derivative.
    name : str
        Name of the derivative dataset.
    bids_version: str
        Version of the BIDS standard.
    exist_ok : bool
        Control the behavior of pathlib.Path.mkdir when a derivative folder
        with this name already exists.
    propagate: bool
        If set to True (default to False), fields that are not explicitly
        provided in desc_kwargs get propagated to the derivatives. Else,
        these fields get no values.
    desc_kwargs: dict
        Dictionary of entries that should be added to the
        dataset_description.json file.
    """
    source_dir = Path(source_dir)

    deriv_dir = source_dir / "derivatives" / name

    desc = {
        'Name': name,
        'BIDSVersion': bids_version,
        'PipelineDescription': {
            "Name": name
            }
        }

    fname = source_dir / 'dataset_description.json'
    if not fname.exists():
        raise ValueError("The argument source_dir must point to a valid BIDS directory." +
                         "As such, it should contain a dataset_description.json file.")
    orig_desc = json.loads(fname.read_text())

    if propagate:
        for field_type in ["recommended", "optional"]:
            for field in get_description_fields(bids_version, field_type):
                if field in desc:
                    continue
                if field in orig_desc:
                    desc[field] = orig_desc[field]

    desc.update(desc_kwargs)


    for field in get_description_fields(bids_version, "required"):
        if field not in desc:
            raise ValueError("The field {} is required and is currently missing.".format(field))

    deriv_dir.mkdir(parents=True, exist_ok=exist_ok)
    Path.write_text(deriv_dir / 'dataset_description.json', json.dumps(desc, indent=4))
