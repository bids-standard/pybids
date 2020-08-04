"""Miscellaneous layout-related utilities."""
import os
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
        if not os.path.exists(path):
            raise ConfigError(
                'Configuration file "{}" does not exist'.format(k))
        if k in cf.get_option('config_paths'):
            raise ConfigError('Configuration {!r} already exists'.format(k))

    kwargs.update(**cf.get_option('config_paths'))
    cf.set_option('config_paths', kwargs)
