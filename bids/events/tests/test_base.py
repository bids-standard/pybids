from bids.events.base import (SparseBIDSColumn, BIDSEventCollection,
                              BIDSEventFile)
import pytest
from os.path import join, dirname, abspath, exists
import os
from bids import grabbids
import tempfile
import pandas as pd
from glob import glob
import shutil

@pytest.fixture
def bids_event_collection():
    mod_file = abspath(grabbids.__file__)
    path = join(dirname(mod_file), 'tests', 'data', 'ds005')
    return BIDSEventCollection(path)


def test_collection(bids_event_collection):
    ''' Integration test for BIDSEventCollection initialization. '''

    bec = bids_event_collection
    bec.read()

    # Test collection attributes
    assert bec.condition_column == 'trial_type'

    # Test that event files are loaded properly
    assert len(bec.event_files) == 48
    ef = bec.event_files[0]
    assert isinstance(ef, BIDSEventFile)
    assert ef.entities['task'] == 'mixedgamblestask'
    assert ef.entities['subject'] == '01'

    # Test extracted columns
    col_keys = bec.columns.keys()
    assert set(col_keys) == {'RT', 'gain', 'respnum', 'PTval', 'loss',
                             'respcat', 'parametric gain',
                             'trial_type/parametric gain'}
    col = bec.columns['RT']
    assert isinstance(col, SparseBIDSColumn)
    assert col.collection == bec
    assert col.name == 'RT'
    assert col.onsets.max() == 476
    assert (col.durations == 3).all()
    assert len(col.durations) == 4096
    ents = col.entities
    assert (ents['task'] == 'mixedgamblestask').all()
    assert set(ents.columns) == {'task', 'subject', 'run', 'event_file_id'}


def test_read_from_files(bids_event_collection):
    bec = bids_event_collection
    path = join(dirname(abspath(grabbids.__file__)), 'tests', 'data', 'ds005')
    subs = ['02', '06', '08']
    template = 'sub-%s/func/sub-%s_task-mixedgamblestask_run-01_events.tsv'
    files = [join(path, template % (s, s)) for s in subs]
    # Put them in a temporary directory
    tmp_dir = tempfile.mkdtemp()
    for f in files:
        shutil.copy2(f, tmp_dir)

    bec.read(file_directory=tmp_dir)
    col_keys = bec.columns.keys()
    assert set(col_keys) == {'RT', 'gain', 'respnum', 'PTval', 'loss',
                             'respcat', 'parametric gain',
                             'trial_type/parametric gain'}
    shutil.rmtree(tmp_dir)


def test_write_collection(bids_event_collection):

    # TODO: test content of files, not just existence
    bec = bids_event_collection
    bec.read()

    # Sparse, single file
    filename = tempfile.mktemp() + '.tsv'
    bec.write(file=filename, sparse=True)
    assert exists(filename)
    os.remove(filename)

    # Dense, single file
    bec.write(file=filename, sparse=False, sampling_rate=1)
    assert exists(filename)
    os.remove(filename)

    # Sparse, one output file per input file
    tmpdir = tempfile.mkdtemp()
    bec.write(tmpdir, sparse=True)
    files = glob(join(tmpdir, "*.tsv"))
    assert len(files) == bec.dense_index['event_file_id'].nunique()
    shutil.rmtree(tmpdir)

    # Dense, one output file per input file
    tmpdir = tempfile.mkdtemp()
    bec.write(tmpdir, sparse=False)
    files = glob(join(tmpdir, "*.tsv"))
    assert len(files) == bec.dense_index['event_file_id'].nunique()
    shutil.rmtree(tmpdir)
