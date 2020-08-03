"""Test miscellaneous utilities."""

import os

import pytest
import bids
from bids.exceptions import ConfigError

from ..models import Entity, Config
from ..utils import BIDSMetadata, parse_file_entities, add_config_paths


def test_bidsmetadata_class():
    md = BIDSMetadata("fakefile")
    with pytest.raises(KeyError) as err:
        md["Missing"]
    assert "Metadata term 'Missing' unavailable for file fakefile." in str(err)
    md["Missing"] = 1
    assert md["Missing"] == 1


@pytest.mark.parametrize("extension_initial_dot", (True, False))
def test_parse_file_entities(mock_config, extension_initial_dot):
    filename = '/sub-03_ses-07_run-4_desc-bleargh_sekret.nii.gz'

    dot = '.' if extension_initial_dot else ''

    # Test with entities taken from bids config
    target = {'subject': '03', 'session': '07', 'run': 4, 'suffix': 'sekret',
              'extension': dot + 'nii.gz'}
    assert target == parse_file_entities(filename, config='bids')
    config = Config.load('bids')
    assert target == parse_file_entities(filename, config=[config])

    # Test with entities taken from bids and derivatives config
    target = {'subject': '03', 'session': '07', 'run': 4, 'suffix': 'sekret',
              'desc': 'bleargh', 'extension': dot + 'nii.gz'}
    assert target == parse_file_entities(filename)
    assert target == parse_file_entities(
        filename, config=['bids', 'derivatives'])

    # Test with list of Entities
    entities = [
        Entity('subject', "[/\\\\]sub-([a-zA-Z0-9]+)"),
        Entity('run', "[_/\\\\]run-0*(\\d+)", dtype=int),
        Entity('suffix', "[._]*([a-zA-Z0-9]*?)\\.[^/\\\\]+$"),
        Entity('desc', "desc-([a-zA-Z0-9]+)"),
    ]
    # Leave out session to distinguish from previous test target
    target = {'subject': '03', 'run': 4, 'suffix': 'sekret', 'desc': 'bleargh'}
    assert target == parse_file_entities(filename, entities=entities)


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
