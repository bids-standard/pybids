''' Utilities for manipulating package-level settings. '''

import json
from pathlib import Path
import os
import warnings

from .utils import listify

__all__ = ['set_option', 'set_options', 'get_option']

_config_name = 'pybids_config.json'

conf_path = str(Path(__file__).absolute().parent.joinpath('layout', 'config', '{}.json'))
_default_settings = {
    'config_paths': {
        name: conf_path.format(name) for name in ['bids', 'derivatives']},
    # XXX 0.16: Remove
    'extension_initial_dot': True,
}


def set_option(key, value):
    """ Set a package-wide option.

    Args:
        key (str): The name of the option to set.
        value (object): The new value of the option.
    """
    if key not in _settings:
        raise ValueError("Invalid pybids setting: '%s'" % key)
    # XXX 0.16: Remove
    elif key == "extension_initial_dot":
        if value is not True:
            raise ValueError(f"Cannot set {key!r} to {value!r} as of pybids 0.14. "
                             "This setting is always True, and will be removed "
                             "entirely in 0.16.")
        warnings.warn("Setting 'extension_initial_dot' will be removed in pybids 0.16.",
                      FutureWarning)
    _settings[key] = value


def set_options(**kwargs):
    """ Set multiple package-wide options.

    Args:
        kwargs: Keyword arguments to pass onto set_option().
    """
    for k, v in kwargs.items():
        set_option(k, v)


def get_option(key):
    """ Retrieve the current value of a package-wide option.

    Args:
        key (str): The name of the option to retrieve.

    """
    if key not in _settings:
        raise ValueError("Invalid pybids setting: '%s'" % key)
    return _settings[key]


def from_file(filenames, error_on_missing=True):
    """ Load package-wide settings from specified file(s).

    Args:
        filenames (str, list): Filename or list of filenames containing JSON
            dictionary of settings.
        error_on_missing (bool): If True, raises an error if a file doesn't
            exist.
    """
    filenames = listify(filenames)
    for f in filenames:
        if Path(f).exists():
            settings = json.loads(Path(f).read_text(encoding='utf-8'))
            _settings.update(settings)
        elif error_on_missing:
            raise ValueError("Config file '%s' does not exist." % f)


def reset_options(update_from_file=False):
    """ Reset all options to the package defaults.

    Args:
        update_from_file (bool): If True, re-applies any config files found in
            standard locations.
    """
    global _settings
    _settings = _default_settings.copy()
    if update_from_file:
        _update_from_standard_locations()


def _update_from_standard_locations():
    """ Check standard locations for config files and update settings if found.
    Order is user's home dir, environment variable ($PYBIDS_CONFIG), and then
    current directory--with later files taking precedence over earlier ones.
    """
    locs = [
        Path.home() / _config_name,
        Path('.') / _config_name
    ]
    if 'PYBIDS_CONFIG' in os.environ:
        locs.insert(1, os.environ['PYBIDS_CONFIG'])
    from_file(locs, False)


_settings = {}
reset_options(True)
