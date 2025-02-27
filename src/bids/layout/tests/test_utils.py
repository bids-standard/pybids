"""Test miscellaneous utilities."""

import os

import numpy as np
import pytest
import pandas as pd
import bids
from bids.exceptions import ConfigError

from ..models import Entity, Config
from ..utils import BIDSMetadata, PaddedInt, parse_file_entities, add_config_paths


def test_bidsmetadata_class():
    md = BIDSMetadata("fakefile")
    with pytest.raises(KeyError) as err:
        md["Missing"]
    assert "Metadata term 'Missing' unavailable for file fakefile." in str(err)
    md["Missing"] = 1
    assert md["Missing"] == 1


def test_parse_file_entities(mock_config):
    filename = '/sub-03_ses-07_run-4_desc-bleargh_sekret.nii.gz'

    # Test with entities taken from bids config
    target = {'subject': '03', 'session': '07', 'run': 4, 'suffix': 'sekret',
              'extension': '.nii.gz'}
    assert target == parse_file_entities(filename, config='bids')
    config = Config.load('bids')
    assert target == parse_file_entities(filename, config=[config])

    # Test with entities taken from bids and derivatives config
    target = {'subject': '03', 'session': '07', 'run': 4, 'suffix': 'sekret',
              'desc': 'bleargh', 'extension': '.nii.gz'}
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


@pytest.mark.parametrize(
    "filename, target",
    [
        ('/path/to/sub-01.ext', {'subject': '01', 'extension': '.ext'}),
        ('/path/to/stub.ext', {'suffix': 'stub', 'extension': '.ext'}),
        ('/path/to/.dotfile', {}),
        ('/path/to/stub', {}),
    ]
)
def test_parse_degenerate_files(mock_config, filename, target):
    assert parse_file_entities(filename, config='bids') == target


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


def test_PaddedInt_array_comparisons():
    # Array comparisons should work, not raise exceptions
    arr = np.array([5, 5, 5])
    assert np.all(arr == PaddedInt(5))
    assert np.all(arr == PaddedInt("05"))
    assert np.all(PaddedInt(5) == arr)
    assert np.all(PaddedInt("05") == arr)

    # If the value gets put into an array, it should be considered an int
    # We lose the padding, but it's unlikely we would try to recover it
    # from an array.
    assert np.array([PaddedInt(5)]).dtype == np.int64

    # Verify that we do get some False results
    assert np.array_equal(np.array([4, 5, 6]) == PaddedInt(5), [False, True, False])


def test_PaddedInt_dataframe_behavior():
    # Verify that pandas dataframes are not more special than numpy arrays,
    # as far as PaddedInt is concerned
    df = pd.DataFrame({'a': [5, 5, 5]})
    pidf = pd.DataFrame({'a': [PaddedInt(5)] * 3})
    assert np.all(df['a'] == PaddedInt(5))
    assert np.all(df == pidf)

    assert pidf['a'].dtype is np.dtype('int64')
