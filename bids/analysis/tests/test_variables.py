from bids.analysis.variables import (SparseBIDSColumn, BIDSVariableManager,
                                     BIDSEventFile)
import pytest
from os.path import join, dirname, abspath, exists
import os
from bids import grabbids
import tempfile
from glob import glob
import shutil


@pytest.fixture
def manager():
    mod_file = abspath(grabbids.__file__)
    path = join(dirname(mod_file), 'tests', 'data', 'ds005')
    manager = BIDSVariableManager(path)
    manager.load()
    return manager


def test_manager(manager):
    ''' Integration test for BIDSVariableManager initialization. '''

    # Test that event files are loaded properly
    assert len(manager.event_files) == 48
    ef = manager.event_files[0]
    assert isinstance(ef, BIDSEventFile)
    assert ef.entities['task'] == 'mixedgamblestask'
    assert ef.entities['subject'] == '01'

    # Test extracted columns
    col_keys = manager.columns.keys()
    assert set(col_keys) == {'RT', 'gain', 'respnum', 'PTval', 'loss',
                             'respcat', 'parametric gain', 'trial_type'}
    col = manager.columns['RT']
    assert isinstance(col, SparseBIDSColumn)
    assert col.manager == manager
    assert col.name == 'RT'
    assert col.onsets.max() == 476
    assert (col.durations == 3).all()
    assert len(col.durations) == 4096
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

    manager = BIDSVariableManager([path, tmp_dir])
    manager.load()
    col_keys = manager.columns.keys()
    assert set(col_keys) == {'RT', 'gain', 'respnum', 'PTval', 'loss',
                             'respcat', 'parametric gain', 'trial_type'}
    shutil.rmtree(tmp_dir)


def test_write_manager(manager):

    # TODO: test content of files, not just existence

    # Sparse, single file
    filename = tempfile.mktemp() + '.tsv'
    manager.write(file=filename, sparse=True)
    assert exists(filename)
    os.remove(filename)

    # Sparse, one output file per input file
    tmpdir = tempfile.mkdtemp()
    manager.write(tmpdir, sparse=True)
    files = glob(join(tmpdir, "*.tsv"))
    # assert len(files) == manager.dense_index['event_file_id'].nunique()
    shutil.rmtree(tmpdir)

    # UNCOMMENT NEXT BLOCK ONCE TRANSFORMATIONS ARE RE-ADDED
    # # Dense, single file
    # manager.apply('factor', 'trial_type')
    # manager.write(file=filename, sparse=False, sampling_rate=1)
    # assert exists(filename)
    # os.remove(filename)

    # Dense, one output file per input file
    tmpdir = tempfile.mkdtemp()
    manager.write(tmpdir, sparse=False)
    files = glob(join(tmpdir, "*.tsv"))
    # assert len(files) == manager.dense_index['event_file_id'].nunique()
    shutil.rmtree(tmpdir)


def test_match_columns(manager):
    matches = manager.match_columns('^resp', return_type='columns')
    assert len(matches) == 2
    assert all(isinstance(m, SparseBIDSColumn) for m in matches)


def test_get_design_matrix(manager):
    manager.set_analysis_level('run')
    subs = [str(s).zfill(2) for s in [1, 2, 3, 4, 5, 6]]
    dm = manager.get_design_matrix(columns=['RT', 'parametric gain'],
                                   subject=subs)
    assert dm.shape == (4308, 6)
