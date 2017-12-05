from bids.analysis.variables import (SparseEventColumn, load_variables,
                                     BIDSEventFile)
import pytest
from os.path import join, dirname, abspath, exists
import os
from bids import grabbids
import tempfile
from glob import glob
import shutil


@pytest.fixture
def collection():
    mod_file = abspath(grabbids.__file__)
    path = join(dirname(mod_file), 'tests', 'data', 'ds005')
    return load_variables(path)


def test_clone_collection(collection):
    clone = collection.clone()
    assert clone != collection
    props_same = ['entities', 'default_duration', 'sampling_rate', 'layout']
    for ps in props_same:
        assert getattr(collection, ps) is getattr(clone, ps)

    assert collection.columns.keys() == clone.columns.keys()
    assert collection.columns is not clone.columns
    assert collection.dense_index.equals(clone.dense_index)
    assert collection.dense_index is not clone.dense_index


def test_collection(collection):
    ''' Integration test for BIDSVariableCollection initialization. '''

    # Test that event files are loaded properly
    assert len(collection.event_files) == 48
    ef = collection.event_files[0]
    assert isinstance(ef, BIDSEventFile)
    assert ef.entities['task'] == 'mixedgamblestask'
    assert ef.entities['subject'] == '01'

    # Test extracted columns
    col_keys = collection.columns.keys()
    assert set(col_keys) == {'RT', 'gain', 'respnum', 'PTval', 'loss',
                             'respcat', 'parametric gain', 'trial_type'}
    col = collection.columns['RT']
    assert isinstance(col, SparseEventColumn)
    assert col.collection == collection
    assert col.name == 'RT'
    assert col.onset.max() == 476
    assert (col.duration == 3).all()
    assert len(col.duration) == 4096
    ents = col.entities
    assert (ents['task'] == 'mixedgamblestask').all()
    assert set(ents.columns) == {'task', 'subject', 'run', 'event_file_id'}


def test_read_from_files():

    mod_file = abspath(grabbids.__file__)
    path = join(dirname(mod_file), 'tests', 'data', 'ds005')

    path2 = join(dirname(abspath(grabbids.__file__)), 'tests', 'data', 'ds005')
    subs = ['02', '06', '08']
    template = 'sub-%s/func/sub-%s_task-mixedgamblestask_run-01_events.tsv'
    files = [join(path2, template % (s, s)) for s in subs]
    # Put them in a temporary directory
    tmp_dir = tempfile.mkdtemp()
    for f in files:
        shutil.copy2(f, tmp_dir)

    collection = load_variables([path, tmp_dir])
    col_keys = collection.columns.keys()
    assert set(col_keys) == {'RT', 'gain', 'respnum', 'PTval', 'loss',
                             'respcat', 'parametric gain', 'trial_type'}
    shutil.rmtree(tmp_dir)


# def test_write_collection(collection):

#     # TODO: test content of files, not just existence

#     # Sparse, single file
#     filename = tempfile.mktemp() + '.tsv'
#     collection.write(file=filename, sparse=True)
#     assert exists(filename)
#     os.remove(filename)

#     # Sparse, one output file per input file
#     tmpdir = tempfile.mkdtemp()
#     collection.write(tmpdir, sparse=True)
#     files = glob(join(tmpdir, "*.tsv"))
#     # assert len(files) == collection.dense_index['event_file_id'].nunique()
#     shutil.rmtree(tmpdir)

#     # UNCOMMENT NEXT BLOCK ONCE TRANSFORMATIONS ARE RE-ADDED
#     # # Dense, single file
#     # collection.apply('factor', 'trial_type')
#     # collection.write(file=filename, sparse=False, sampling_rate=1)
#     # assert exists(filename)
#     # os.remove(filename)

#     # Dense, one output file per input file
#     tmpdir = tempfile.mkdtemp()
#     collection.write(tmpdir, sparse=False)
#     files = glob(join(tmpdir, "*.tsv"))
#     # assert len(files) == collection.dense_index['event_file_id'].nunique()
#     shutil.rmtree(tmpdir)


def test_match_columns(collection):
    matches = collection.match_columns('^resp', return_type='columns')
    assert len(matches) == 2
    assert all(isinstance(m, SparseEventColumn) for m in matches)


def test_get_design_matrix(collection):
    subs = [str(s).zfill(2) for s in [1, 2, 3, 4, 5, 6]]
    dm = collection.get_design_matrix(columns=['RT', 'parametric gain'],
                                      groupby=['subject', 'run'],
                                      subject=subs)
    assert dm.shape == (4308, 6)
