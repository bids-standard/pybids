"""Tests of functionality in the models module."""

import sys
import os
import pytest
import copy
import json
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import numpy as np

from bids.layout.models import (BIDSFile, Entity, Tag, Base, Config,
                                FileAssociation, BIDSImageFile)
from bids.tests import get_test_data_path



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


def test_entity_matches(tmpdir):
    filename = "aardvark-4-reporting-for-duty.txt"
    tmpdir.mkdir("tmp").join(filename).write("###")
    f = BIDSFile(os.path.join(str(tmpdir), filename))
    e = Entity('avaricious', r'aardvark-(\d+)')
    result = e.match_file(f)
    assert result == '4'


def test_entity_deepcopy(subject_entity):
    e = subject_entity
    clone = copy.deepcopy(subject_entity)
    for attr in ['name', 'pattern', 'mandatory', 'directory', 'regex']:
        assert getattr(e, attr) == getattr(clone, attr)
    assert e != clone


def test_file_associations():
    session = create_session()
    img = BIDSFile('sub-03/func/sub-03_task-rest_run-2_bold.nii.gz')
    md1 = BIDSFile('sub-03/func/sub-03_task-rest_run-2_bold.json')
    md2 = BIDSFile('task-rest_run-2_bold.json')
    assocs = [
        FileAssociation(src=md1.path, dst=img.path, kind="MetadataFor"),
        FileAssociation(src=img.path, dst=md1.path, kind="MetadataIn"),
        FileAssociation(src=md1.path, dst=md2.path, kind="Child"),
        FileAssociation(src=md2.path, dst=md1.path, kind="Parent"),
        FileAssociation(src=md2.path, dst=img.path, kind="Informs")
    ]
    session.add_all([img, md1, md2] + assocs)
    session.commit()
    assert img._associations == [md1, md2] == img.get_associations()
    assert md2._associations == [md1]
    assert img.get_associations(kind='MetadataFor') == []
    assert img.get_associations(kind='MetadataIn') == [md1]
    results = img.get_associations(kind='MetadataIn', include_parents=True)
    assert set(results) == {md1, md2}


def test_tag_init(sample_bidsfile, subject_entity):
    f, e = sample_bidsfile, subject_entity
    tag = Tag(f, e, 'zzz')
    rep = str(tag)
    assert rep.startswith("<Tag file:") and f.path in rep and 'zzz' in rep


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
    target = {'task', 'acquisition'}
    assert set(ent.name for ent in config.entities.values()) == target
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


def test_bidsfile_get_df_from_tsv_gz(layout_synthetic):
    bf = layout_synthetic.get(suffix='physio', extension='tsv.gz')[0]

    # With onsets
    df1 = bf.get_df()
    df2 = bf.get_df(include_timing=True)
    assert df1.equals(df2)
    assert df1.shape == (1600, 3)
    assert set(df1.columns) == {'onset', 'respiratory', 'cardiac'}
    assert df1.iloc[0, 0] == 0.
    assert df1.iloc[1, 0] - df1.iloc[0, 0] == 0.1

    # With onsets and time shifted
    df3 = bf.get_df(adjust_onset=True)
    assert df1.iloc[:, 1:].equals(df3.iloc[:, 1:])
    assert np.allclose(df3.iloc[:,0], df1.iloc[:, 0] + 22.8)


def test_bidsdatafile_enforces_dtype(layout_synthetic):
    bf = layout_synthetic.get(suffix='participants', extension='tsv')[0]
    df = bf.get_df(enforce_dtypes=False)
    assert df.shape[0] == 5
    assert df.loc[:, 'subject_id'].dtype == int
    assert df.loc[:, 'subject_id'][0] == 1
    df = bf.get_df(enforce_dtypes=True)
    assert df.loc[:, 'subject_id'].dtype == 'O'
    assert df.loc[:, 'subject_id'][0] == '001'
    assert df.loc[:, 'subject_id'][1] == '2'


def test_bidsimagefile_get_image():
    path = "synthetic/sub-01/ses-01/func/sub-01_ses-01_task-nback_run-01_bold.nii.gz"
    path = path.split('/')
    path = os.path.join(get_test_data_path(), *path)
    bf = BIDSImageFile(path)
    assert bf.get_image() is not None
    assert bf.get_image().shape == (64, 64, 64, 64)


def test_bidsjsonfile(layout_synthetic):
    jf = layout_synthetic.get(suffix='bold', extension='json')[0]
    d = jf.get_dict()
    assert isinstance(d, dict)
    assert d['RepetitionTime'] == 2.5
    j = jf.get_json()
    assert isinstance(j, str)
    assert 'RepetitionTime' in j
    assert json.loads(j) == d


def test_bidsfile_get_metadata(layout_synthetic):
    bf = layout_synthetic.get(suffix='physio', extension='tsv.gz')[0]
    md = bf.get_metadata()
    assert set(md.keys()) == {'Columns', 'SamplingFrequency', 'StartTime'}


def test_bidsfile_get_entities(layout_synthetic):
    md_ents = {'Columns', 'SamplingFrequency', 'StartTime'}
    file_ents = {'datatype', 'extension', 'run', 'session', 'subject',
                 'suffix', 'task'}
    bf = layout_synthetic.get(suffix='physio', extension='tsv.gz')[10]
    # metadata=True and values='tags'; this is equivalent to get_metadata()
    md = bf.get_entities(metadata=True)
    assert md == bf.get_metadata()
    assert set(md.keys()) == md_ents
    assert md['StartTime'] == 22.8
    # metadata=True and values='objects'
    md = bf.get_entities(metadata=True, values='obj')
    assert set(md.keys()) == md_ents
    assert all([isinstance(v, Entity) for v in md.values()])
    # metadata=False and values='tags'
    md = bf.get_entities(metadata=False, values='tags')
    assert set(md.keys()) == file_ents
    assert md['session'] == '02'
    assert md['task'] == 'nback'
    # metadata=False and values='obj'
    md = bf.get_entities(metadata=False, values='objects')
    assert set(md.keys()) == file_ents
    assert all([isinstance(v, Entity) for v in md.values()])
    # No metadata constraint
    md = bf.get_entities(metadata='all')
    md2 = bf.get_entities(metadata=None)
    assert md == md2
    assert set(md.keys()) == md_ents | file_ents


@pytest.mark.xfail(sys.version_info < (3, 6), reason="os.PathLike introduced in Python 3.6")
def test_bidsfile_fspath(sample_bidsfile):
    bf = sample_bidsfile
    bf_path = Path(bf)
    assert bf_path == Path(bf.path)
    assert bf_path.read_text() == '###'
