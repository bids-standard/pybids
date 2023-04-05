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


class PaddedInt(int):
    """ Integer type that preserves zero-padding

    Acts like an int in almost all ways except that string formatting
    will keep the original zero-padding. Numeric format specifiers

    >>> PaddedInt(1)
    1
    >>> p2 = PaddedInt("02")
    >>> p2
    02
    >>> str(p2)
    '02'
    >>> p2 == 2
    True
    >>> p2 in range(3)
    True
    >>> f"{p2}"
    '02'
    >>> f"{p2:s}"
    '02'
    >>> f"{p2!s}"
    '02'
    >>> f"{p2!r}"
    '02'
    >>> f"{p2:d}"
    '2'
    >>> f"{p2:03d}"
    '002'
    >>> f"{p2:f}"
    '2.000000'
    >>> {2: "val"}.get(p2)
    'val'
    >>> {p2: "val"}.get(2)
    'val'

    Note that arithmetic will break the padding.

    >>> str(p2 + 1)
    '3'
    """
    def __init__(self, val):
        self.sval = str(val)

    def __eq__(self, val):
        return val == self.sval or super().__eq__(val)

    def __str__(self):
        return self.sval

    def __repr__(self):
        return self.sval

    def __format__(self, format_spec):
        """ Format a padded integer

        If a format spec can be used on a string, apply it to the zero-padded string.
        Otherwise format as an integer.
        """
        try:
            return format(self.sval, format_spec)
        except ValueError:
            return super().__format__(format_spec)

    def __hash__(self):
        return super().__hash__()


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
