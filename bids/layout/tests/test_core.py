import pytest
import os
import copy

from bids.layout.models import Config, BIDSFile, Entity
from bids.layout.index import BIDSRootNode
from bids import BIDSLayout
from bids.tests import get_test_data_path


DIRNAME = os.path.dirname(__file__)


@pytest.fixture
def sample_bidsfile(tmpdir):
    testfile = 'sub-03_ses-2_task-rest_acq-fullbrain_run-2_bold.nii.gz'
    fn = tmpdir.mkdir("tmp").join(testfile)
    fn.write('###')
    return BIDSFile(os.path.join(str(fn)))


@pytest.fixture(scope='module')
def subject_entity():
    return Entity('subject', "[/\\\\]sub-([a-zA-Z0-9]+)", False,
               "{subject}", None, bleargh=True)


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


def test_entity_init_minimal():
    e = Entity('avaricious', r'aardvark-(\d+)')
    assert e.name == 'avaricious'
    assert e.pattern == r'aardvark-(\d+)'
    assert not e.mandatory
    assert e.directory is None
    assert e.files == {}


def test_entity_init_all_args(subject_entity):
    ent = subject_entity
    assert ent.name == 'subject'
    assert ent.pattern == "[/\\\\]sub-([a-zA-Z0-9]+)"
    assert ent.mandatory == False
    assert ent.directory == "{subject}"
    assert ent.map_func is None
    assert ent.kwargs == {'bleargh': True}


def test_entity_init_with_bad_dtype():
    with pytest.raises(ValueError) as exc:
        ent = Entity('test', dtype='superfloat')
        msg = exc.value.message
        assert msg.startswith("Invalid dtype")


def test_entity_deepcopy(subject_entity):
    e = subject_entity
    clone = copy.deepcopy(subject_entity)
    for attr in ['name', 'pattern', 'mandatory', 'directory', 'map_func',
                 'regex', 'kwargs']:
        assert getattr(e, attr) == getattr(clone, attr)
    assert e != clone


def test_entity_matches(tmpdir):
    filename = "aardvark-4-reporting-for-duty.txt"
    tmpdir.mkdir("tmp").join(filename).write("###")
    f = BIDSFile(os.path.join(str(tmpdir), filename))
    e = Entity('avaricious', r'aardvark-(\d+)')
    result = e.match_file(f)
    assert result == '4'


def test_entity_matches_with_map_func(sample_bidsfile):
    bf = sample_bidsfile
    e = Entity('test', map_func=lambda x: x.filename.split('-')[1])
    assert e.match_file(bf) == '03_ses'


def test_entity_unique_and_count():
    e = Entity('prop', r'-(\d+)')
    e.files = {
        'test1-10.txt': '10',
        'test2-7.txt': '7',
        'test3-7.txt': '7'
    }
    assert sorted(e.unique()) == ['10', '7']
    assert e.count() == 2
    assert e.count(files=True) == 3


def test_entity_add_file():
    e = Entity('prop', r'-(\d+)')
    e.add_file('a', '1')
    assert e.files['a'] == '1'


def test_bidsfile_init():
    path = '/tmp/sub-03_ses-2_task-rest_acq-pfc_run-2_bold.nii.gz'
    bf = BIDSFile(path)
    assert bf.path == path
    assert bf.dirname == os.path.dirname(path)
    assert bf.filename == os.path.basename(path)
    assert bf.tags == []
    assert bf.entities == {}
    assert bf.parent is None


def test_bidsfile_matches(sample_bidsfile):
    'sub-03_ses-2_task-rest_acq-fullbrain_run-2_bold.nii.gz'
    bf = sample_bidsfile
    bf.entities = {
        'subject': '03',
        'ses': '2',
        'task': 'rest',
        'run': 2,
        'suffix': 'bold'
    }

    # Plain Jane matching
    entities = { 'subject': '03', 'task': 'rest', 'suffix': 'bold'}
    assert bf._matches(entities)
    assert bf._matches(entities, regex_search=True)

    # Regex matching
    regex_ents = { 'subject': r'\d+', 'task': 're.t', 'suffix': '.*ld'}
    assert not bf._matches(regex_ents)
    assert bf._matches(regex_ents, regex_search=True)

    # Mis-matching entities
    bad_ents = entities.copy()
    bad_ents['suffix'] = 'no match'
    assert not bf._matches(bad_ents)

    # Missing leading 0s should match regardless of regex status
    mod_ents = entities.copy()
    mod_ents['subject'] = 3
    assert bf._matches(mod_ents)
    assert bf._matches(mod_ents, regex_search=True)

    # Lists can be passed
    mod_ents = entities.copy()
    mod_ents['subject'] = [3, '3']
    assert bf._matches(mod_ents)
    assert bf._matches(mod_ents, regex_search=True)
    mod_ents['subject'] = ['3', 2, r'\d+']
    assert not bf._matches(mod_ents)
    assert bf._matches(mod_ents, regex_search=True)


def test_bidsfile_image_property():
    path = os.path.join(get_test_data_path(), 'synthetic', 'sub-01', 'ses-01',
                        'func', 'sub-01_ses-01_task-nback_run-01_bold.nii.gz')
    bf = BIDSFile(path)
    img = bf.image
    assert img.__class__.__name__ == 'Nifti1Image'
    assert img.header.get_data_shape() == (64, 64, 64, 64)
