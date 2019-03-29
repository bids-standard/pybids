import os
import pytest
import bids
from bids.layout.models import BIDSFile, Entity, Tag, Base, Config, Scope
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture(scope="module")
def session():
    engine = create_engine('sqlite://')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()

@pytest.fixture
def sample_bidsfile(tmpdir):
    testfile = 'sub-03_ses-2_task-rest_acq-fullbrain_run-2_bold.nii.gz'
    fn = tmpdir.mkdir("tmp").join(testfile)
    fn.write('###')
    return BIDSFile(os.path.join(str(fn)))


@pytest.fixture(scope='module')
def subject_entity():
    return Entity('subject', r"[/\\\\]sub-([a-zA-Z0-9]+)", mandatory=False,
               directory="{subject}", dtype='str')


def test_entity_initialization():
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


def test_tag_dtype(sample_bidsfile, subject_entity):
    f, e = sample_bidsfile, subject_entity
    # Various ways of initializing--should all give same result
    tags = [
        Tag(f, e, 4, int),
        Tag(f, e, '4', 'int'),
        Tag(f, e, '4', int),
        Tag(f, e, 4),
        Tag(file=f, entity=e, dtype=int, value='4')
    ]
    assert all([t.dtype == int for t in tags])


def test_entity_add_file(sample_bidsfile, session):
    bf = sample_bidsfile
    e = Entity('prop', r'-(\d+)')
    t = Tag(file=bf, entity=e, value=4)
    session.add(t)
    session.commit()
    print(e.files)
    print(bf.entities)
    assert e.files[bf.path] == '4'


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
    assert [ent.name for ent in config.entities] == ['task', 'acquisition']
    assert config.default_path_patterns  == patterns


def test_scope_init():
    scope = Scope(name='bids', path='/tmp')
    assert scope.name == 'bids'
    assert scope.path == '/tmp'