''' Utilities for manipulating package-level settings. '''

import json
from os.path import join, expanduser, exists
import os
from io import open
import warnings

__all__ = ['set_option', 'set_options', 'get_option']

_config_name = 'pybids_config.json'

conf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'layout', 'config', '{}.json')
_default_settings = {
    # XXX 0.14: Remove bids-nodot option (and file)
    'config_paths': {
        name: conf_path.format(name) for name in ['bids', 'derivatives', 'bids-nodot']},
    # XXX 0.14: Set to True
    'extension_initial_dot': None,
}


def set_option(key, value):
    """ Set a package-wide option.

    Args:
        key (str): The name of the option to set.
        value (object): The new value of the option.
    """
    if key not in _settings:
        raise ValueError("Invalid pybids setting: '%s'" % key)
    # XXX 0.14: Raise error
    if (key, value) == ("extension_initial_dot", False):
        warnings.warn("Setting 'extension_initial_dot' to False will be disabled in "
                      "pybids 0.14", FutureWarning)
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
    if isinstance(filenames, str):
        filenames = [filenames]
    for f in filenames:
        if exists(f):
            with open(f, 'r', encoding='utf-8') as fobj:
                settings = json.load(fobj)
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
        join(expanduser('~'), _config_name),
        join('.', _config_name)
    ]
    if 'PYBIDS_CONFIG' in os.environ:
        locs.insert(1, os.environ['PYBIDS_CONFIG'])
    from_file(locs, False)


_settings = {}
reset_options(True)
