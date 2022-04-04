"""Test miscellaneous utilities."""

import os

import pytest

from pathlib import Path

import bids

from bids.exceptions import ConfigError

from ..models import Entity, Config

from ..utils import BIDSMetadata, parse_file_entities, add_config_paths
from ..utils import write_description
from ..utils import get_description_fields

from ...tests import get_test_data_path




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

# teardown
# @pytest.fixture()
# def clean_up(output_dir):
#     yield
#     os.remove(output_dir)

def test_write_description_raw(exist_ok=True):

    write_description(name="test", is_derivative=False, exist_ok=exist_ok)

    output_dir = Path().resolve();
    bids.BIDSLayout(Path.joinpath(output_dir, 'raw'))

    # teardown
    os.remove(Path.joinpath(output_dir, 'raw', 'dataset_description.json'))

def test_write_description_derivatives(exist_ok=True):

    source_dir = get_test_data_path("Path") / '7t_trt'

    write_description(source_dir=source_dir, name="test", 
                                exist_ok=exist_ok)

    bids.BIDSLayout(source_dir, derivatives=True)

    # teardown
    os.remove(Path.joinpath(source_dir, 'derivatives', 'test', 'dataset_description.json'))

def test_write_description_derivatives_outside_raw(exist_ok=True):

    source_dir = get_test_data_path("Path") / '7t_trt'
    output_dir = Path().resolve();

    write_description(source_dir=source_dir, name="test",
                      output_dir=output_dir,
                      exist_ok=exist_ok)

    bids.BIDSLayout(Path.joinpath(output_dir, 'derivatives', 'test'))

    # teardown
    os.remove(Path.joinpath(output_dir, 'derivatives', 'test', 'dataset_description.json'))


def test_get_description_fields():

    fields = get_description_fields("1.1.1", "required")
    assert fields == ["Name", "BIDSVersion"]

    fields = get_description_fields("1.6.1", "required")
    assert fields == ["Name", "BIDSVersion", "GeneratedBy"]

def test_get_description_fields_error():

    pytest.raises(TypeError, get_description_fields, 1, "required")