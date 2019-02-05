import pytest
from bids.layout.core import Config, BIDSFile, Entity, BIDSRootNode
from bids import BIDSLayout
import os
from os.path import join
import posixpath as psp
import tempfile
import json
from copy import copy


DIRNAME = os.path.dirname(__file__)


@pytest.fixture
def file(tmpdir):
    testfile = 'sub-03_ses-2_task-rest_acq-fullbrain_run-2_bold.nii.gz'
    fn = tmpdir.mkdir("tmp").join(testfile)
    fn.write('###')
    return BIDSFile(join(str(fn)))


def test_config_init_bare():
    config = Config('custom')
    assert config.name == 'custom'
    assert config.entities == {}
    assert config.default_path_patterns is None


def test_config_init_with_args():
    ents = [
        {
            "name": "task",
            "pattern": "[_/\\\\]task-([a-zA-Z0-9]+)"
        },
        {
            "name": "acquisition",
            "pattern": "[_/\\\\]acq-([a-zA-Z0-9]+)"
        }
    ]
    patterns = ['this_will_never_match_anything', 'and_neither_will_this']
    config = Config('custom', entities=ents, default_path_patterns=patterns)
    assert config.name == 'custom'
    assert {'task', 'acquisition'} == set(config.entities.keys())
    assert config.default_path_patterns  == patterns


def test_config_init_from_class_load_bids():
    config = Config.load('bids')
    assert config.name == 'bids'
    ent_names = ('subject', 'run', 'suffix')
    assert all([en in config.entities for en in ent_names])
    assert 'space' not in config.entities
    first_patt = 'sub-{subject}[/ses-{session}]/anat/sub-{subject}'
    assert config.default_path_patterns[0].startswith(first_patt)


def test_config_init_from_class_load_derivatives():
    config = Config.load('derivatives')
    assert config.name == 'derivatives'
    ent_names = ('space', 'atlas', 'roi')
    assert all([en in config.entities for en in ent_names])
    assert 'subject' not in config.entities
    assert config.default_path_patterns is None


