"""Test miscellaneous utilities."""

import os

import pytest
import bids
from bids.exceptions import ConfigError

from ..utils import BIDSMetadata, add_config_paths


def test_bidsmetadata_class():
    md = BIDSMetadata("fakefile")
    with pytest.raises(KeyError) as err:
        md["Missing"]
    assert "Metadata term 'Missing' unavailable for file fakefile." in str(err)
    md["Missing"] = 1
    assert md["Missing"] == 1

def test_add_config_paths():
    bids_dir = os.path.dirname(bids.__file__)
    bids_json = os.path.join(bids_dir, 'layout', 'config', 'bids.json')
    with pytest.raises(ConfigError) as exc:
        add_config_paths(test_config1='nonexistentpath.json')
    assert str(exc.value).startswith('Configuration file')
    with pytest.raises(ConfigError) as exc:
        add_config_paths(bids=bids_json)
    assert str(exc.value).startswith("Configuration 'bids' already")
    add_config_paths(dummy=bids_json)
    config = Config.load('dummy')
    assert 'subject' in config.entities
