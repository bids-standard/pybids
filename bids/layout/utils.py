"""Miscellaneous layout-related utilities."""
from pathlib import Path

from .. import config as cf
from ..utils import make_bidsfile, listify
from ..exceptions import ConfigError


class BIDSMetadata(dict):
    """ Metadata dictionary that reports the associated file on lookup failures. """
    def __init__(self, source_file):
        self._source_file = source_file
        super().__init__()

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError as e:
            raise KeyError(
                "Metadata term {!r} unavailable for file {}.".format(key, self._source_file))


def parse_file_entities(filename, entities=None, config=None,
                        include_unmatched=False):
    """Parse the passed filename for entity/value pairs.

    Parameters
    ----------
    filename : str
        The filename to parse for entity values
    entities : list or None, optional
        An optional list of Entity instances to use in extraction.
        If passed, the config argument is ignored. Default is None.
    config : str or :obj:`bids.layout.models.Config` or list or None, optional
        One or more :obj:`bids.layout.models.Config` objects or names of
        configurations to use in matching. Each element must be a
        :obj:`bids.layout.models.Config` object, or a valid
        :obj:`bids.layout.models.Config` name (e.g., 'bids' or 'derivatives').
        If None, all available configs are used. Default is None.
    include_unmatched : bool, optional
        If True, unmatched entities are included in the returned dict,
        with values set to None.
        If False (default), unmatched entities are ignored.

    Returns
    -------
    dict
        Keys are Entity names and values are the values from the filename.
    """
    # Load Configs if needed
    if entities is None:

        if config is None:
            config = ['bids', 'derivatives']

        from .models import Config
        config = [Config.load(c) if not isinstance(c, Config) else c
                  for c in listify(config)]

        # Consolidate entities from all Configs into a single dict
        entities = {}
        for c in config:
            entities.update(c.entities)
        entities = entities.values()

    # Extract matches
    bf = make_bidsfile(filename)
    ent_vals = {}
    for ent in entities:
        match = ent.match_file(bf)
        if match is not None or include_unmatched:
            ent_vals[ent.name] = match

    return ent_vals


def add_config_paths(**kwargs):
    """Add to the pool of available configuration files for BIDSLayout.

    Parameters
    ----------
    kwargs : dict
        Dictionary specifying where to find additional config files.
        Keys are names, values are paths to the corresponding .json file.

    Examples
    --------
    > add_config_paths(my_config='/path/to/config')
    > layout = BIDSLayout('/path/to/bids', config=['bids', 'my_config'])
    """
    for k, path in kwargs.items():
        if not Path(path).exists():
            raise ConfigError(
                'Configuration file "{}" does not exist'.format(k))
        if k in cf.get_option('config_paths'):
            raise ConfigError('Configuration {!r} already exists'.format(k))

    kwargs.update(**cf.get_option('config_paths'))
    cf.set_option('config_paths', kwargs)

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
