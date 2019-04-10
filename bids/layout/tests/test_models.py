import os
import pytest
import bids
from bids.layout.models import BIDSFile, Entity, Tag, Base, Config
from bids.layout import BIDSLayout
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def create_session():
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
    assert ent.pattern == r"[/\\\\]sub-([a-zA-Z0-9]+)"
    assert ent.mandatory == False
    assert ent.directory == "{subject}"


def test_entity_init_with_bad_dtype():
    with pytest.raises(ValueError) as exc:
        ent = Entity('test', dtype='superfloat')
        msg = exc.value.message
        assert msg.startswith("Invalid dtype")


# def test_entity_deepcopy(subject_entity):
#     e = subject_entity
#     clone = copy.deepcopy(subject_entity)
#     for attr in ['name', 'pattern', 'mandatory', 'directory',
#                  'regex', 'kwargs']:
#         assert getattr(e, attr) == getattr(clone, attr)
#     assert e != clone


def test_entity_matches(tmpdir):
    filename = "aardvark-4-reporting-for-duty.txt"
    tmpdir.mkdir("tmp").join(filename).write("###")
    f = BIDSFile(os.path.join(str(tmpdir), filename))
    e = Entity('avaricious', r'aardvark-(\d+)')
    result = e.match_file(f)
    assert result == '4'


# def test_entity_unique_and_count():
#     e = Entity('prop', r'-(\d+)')
#     e.files = {
#         'test1-10.txt': '10',
#         'test2-7.txt': '7',
#         'test3-7.txt': '7'
#     }
#     assert sorted(e.unique()) == ['10', '7']
#     assert e.count() == 2
#     assert e.count(files=True) == 3


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


def test_entity_add_file(sample_bidsfile):
    session = create_session()
    bf = sample_bidsfile
    e = Entity('prop', r'-(\d+)')
    t = Tag(file=bf, entity=e, value=4)
    session.add_all([t, e, bf])
    session.commit()
    assert e.files[bf.path] == 4


def test_config_init_with_args():
    session = create_session()
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
    assert [ent.name for ent in config.entities.values()] == ['task', 'acquisition']
    assert config.default_path_patterns  == patterns


def test_load_existing_config():
    session = create_session()
    first = Config('dummy')
    session.add(first)
    session.commit()

    second = Config.load({"name": "dummy"}, session=session)
    assert first == second
    session.add(second)
    session.commit()

    from sqlalchemy.orm.exc import FlushError
    with pytest.raises(FlushError):
        second = Config.load({"name": "dummy"})
        session.add(second)
        session.commit()
