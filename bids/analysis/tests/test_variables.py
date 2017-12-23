from bids.grabbids import BIDSLayout
from bids.analysis.variables import (SparseEventColumn, load_variables,
                                     BIDSRunInfo, load_event_variables)
import pytest
from os.path import join, dirname, abspath
from bids import grabbids
# from grabbit import merge_layouts
import tempfile
import shutil


@pytest.fixture
def collection():
    mod_file = abspath(grabbids.__file__)
    path = join(dirname(mod_file), 'tests', 'data', 'ds005')
    layout = BIDSLayout(path)
    return load_event_variables(layout, scan_length=480)


# def test_load_all_bids_variables():
#     mod_file = abspath(grabbids.__file__)
#     path = join(dirname(mod_file), 'tests', 'data', '7t_trt')
#     layout = BIDSLayout(path)
#     # collection = load_variables(layout, acq='fullbrain')
#     # TODO


def test_load_dense_variables():
    mod_file = abspath(grabbids.__file__)
    path = join(dirname(mod_file), 'tests', 'data', '7t_trt')
    layout = BIDSLayout(path)
    load_event_variables(layout)


def test_aggregate_column(collection):
    col = collection['RT']
    agg_col = col.aggregate('subject')
    assert agg_col.values.shape[0] == 16
    assert agg_col.entities.shape[0] == 16
    assert set(agg_col.entities.columns) == {'subject'}
    agg_col2 = col.aggregate('subject', func='max')
    assert (agg_col2.values >= agg_col.values).all() and \
        agg_col2.values.mean() > agg_col.values.mean()

    # Dense column
    dc = col.to_dense()
    agg_col = dc.aggregate('subject')
    assert agg_col.values.shape[0] == 16
    assert agg_col.entities.shape[0] == 16
    assert set(agg_col.entities.columns) == {'subject'}
    agg_col2 = dc.aggregate('subject', func='max')
    assert (agg_col2.values >= agg_col.values).all() and \
        agg_col2.values.mean() > agg_col.values.mean()


def test_clone_collection(collection):
    clone = collection.clone()
    assert clone != collection
    props_same = ['entities', 'sampling_rate']
    for ps in props_same:
        assert getattr(collection, ps) is getattr(clone, ps)

    assert collection.columns.keys() == clone.columns.keys()
    assert collection.columns is not clone.columns
    assert collection.dense_index.equals(clone.dense_index)
    assert collection.dense_index is not clone.dense_index


def test_collection(collection):
    ''' Integration test for BIDSVariableCollection initialization. '''

    # Test that event files are loaded properly
    assert len(collection.run_infos) == 48
    ef = collection.run_infos[0]
    assert isinstance(ef, BIDSRunInfo)
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
    assert set(ents.columns) == {'task', 'subject', 'run', 'unique_run_id'}


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

    layout = BIDSLayout(path)
    # layout2 = BIDSLayout(tmp_dir)
    # layout = merge_layouts([layout, layout2])

    # Time-level variables
    collection = load_event_variables(layout, scan_length=480)
    col_keys = collection.columns.keys()
    assert set(col_keys) == {'RT', 'gain', 'respnum', 'PTval', 'loss',
                             'respcat', 'parametric gain', 'trial_type'}

    # Subject-level variables
    collection = load_variables(layout, 'subject')
    col_keys = collection.columns.keys()
    assert set(col_keys) == {'sex', 'age'}
    shutil.rmtree(tmp_dir)


def test_match_columns(collection):
    matches = collection.match_columns('^resp', return_type='columns')
    assert len(matches) == 2
    assert all(isinstance(m, SparseEventColumn) for m in matches)


def test_get_design_matrix(collection):
    sub_ids = [1, 2, 3, 4, 5, 6]
    subs = [str(s).zfill(2) for s in sub_ids]
    dm = collection.get_design_matrix(columns=['RT', 'parametric gain'],
                                      groupby=['subject', 'run'],
                                      subject=subs)
    assert set(dm['subject'].unique()) == set(sub_ids)
    cols = set(['amplitude', 'onset', 'duration', 'subject', 'run', 'task',
                'condition'])
    assert set(dm.columns) == cols
    assert set(dm['condition'].unique()) == {'RT', 'parametric gain'}
    assert dm.shape == (3072, 7)
