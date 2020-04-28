from os.path import join, dirname, abspath

import pytest

from bids.layout import BIDSLayout
from bids.tests import get_test_data_path
from bids.variables import (DenseRunVariable, SparseRunVariable,
                            merge_collections)
from bids.variables.entities import RunInfo

@pytest.fixture(scope="module")
def run_coll():
    path = join(get_test_data_path(), 'ds005')
    layout = BIDSLayout(path)
    # Limit to a few subjects to reduce test running time
    return layout.get_collections('run', types=['events'], merge=True,
                                  scan_length=480, subject=['01', '02', '04'])


@pytest.fixture(scope="module")
def run_coll_list():
    path = join(get_test_data_path(), 'ds005')
    layout = BIDSLayout(path)
    return layout.get_collections('run', types=['events'], merge=False,
                                  scan_length=480, subject=['01', '02', '04'])


def test_run_variable_collection_init(run_coll):
    assert isinstance(run_coll.variables, dict)
    assert run_coll.sampling_rate == 10


def test_run_variable_collection_sparse_variable_accessors(run_coll):
    coll = run_coll.clone()
    assert coll.get_sparse_variables()
    assert coll.all_sparse()
    coll.variables['RT'] = coll.variables['RT'].to_dense(1)
    assert not coll.all_sparse()
    assert len(coll.get_sparse_variables()) + 1 == len(coll.variables)


def test_run_variable_collection_dense_variable_accessors(run_coll):
    coll = run_coll.clone()
    coll.variables['RT'] = coll.variables['RT'].to_dense(1)
    assert not coll.all_dense()
    assert len(coll.get_dense_variables()) == 1
    for k, v in coll.variables.items():
        if k == 'RT':
            continue
        coll.variables[k] = v.to_dense(1)
    assert coll.all_dense()


def test_run_variable_collection_get_sampling_rate(run_coll):
    coll = run_coll.clone()
    assert coll._get_sampling_rate(None) == 10
    assert coll._get_sampling_rate('TR') == 0.5
    coll.variables['RT'].run_info[0] = RunInfo({}, 200, 10, None)
    with pytest.raises(ValueError) as exc:
        coll._get_sampling_rate('TR')
        assert exc.value.message.startswith('Non-unique')
    assert coll._get_sampling_rate('highest') == 10
    coll.variables['RT1'] = coll.variables['RT'].to_dense(5.)
    coll.variables['RT2'] = coll.variables['RT'].to_dense(12.)
    assert coll._get_sampling_rate('highest') == 12.
    assert coll._get_sampling_rate(20) == 20
    with pytest.raises(ValueError) as exc:
        coll._get_sampling_rate('BLARGH')
        assert exc.value.message.startswith('Invalid')


def test_resample_run_variable_collection(run_coll):
    run_coll = run_coll.clone()
    resampled = run_coll.resample()
    assert not resampled.variables  # Empty because all variables are sparse

    resampled = run_coll.resample(force_dense=True).variables
    assert len(resampled) == 7
    assert all([isinstance(v, DenseRunVariable) for v in resampled.values()])
    assert len(set([v.sampling_rate for v in resampled.values()])) == 1
    targ_len = 480 * 3 * 3 * 10
    assert all([len(v.values) == targ_len for v in resampled.values()])

    sr = 20
    resampled = run_coll.resample(sr, force_dense=True).variables
    targ_len = 480 * 3 * 3 * sr
    assert all([len(v.values) == targ_len for v in resampled.values()])

    run_coll.resample(sr, force_dense=True, in_place=True)
    assert len(run_coll.variables) == 8
    vars_ = run_coll.variables.values()
    vars_ = [v for v in vars_ if v.name != 'trial_type']
    assert all([len(v.values) == targ_len for v in vars_])
    assert all([v.sampling_rate == sr for v in vars_])
    assert all([isinstance(v, DenseRunVariable) for v in vars_])


def test_run_variable_collection_to_df_all_sparse_vars(run_coll):
    run_coll = run_coll.clone()

    timing_cols = {'onset', 'duration'}
    entity_cols = {'subject', 'run', 'task',  'suffix', 'datatype'}
    cond_names = {'PTval', 'RT', 'gain', 'loss', 'parametric gain', 'respcat',
                  'respnum', 'trial_type'}
    condition = {'condition'}
    ampl = {'amplitude'}

    # Fails because all vars are sparse
    with pytest.raises(ValueError):
        df = run_coll.to_df(include_sparse=False)

    # Fails because no such variables exist
    with pytest.raises(ValueError):
        df = run_coll.to_df(variables=['rubadubdub', '999'])

    # Wide format
    df = run_coll.to_df()
    events_per_sub = 256
    assert df.shape == (events_per_sub * 3, 15) 
    assert set(df.columns) == timing_cols.union(entity_cols, cond_names)

    # Wide format, selecting variables by name
    df = run_coll.to_df(format='wide', variables=['RT', 'PTval'])
    assert df.shape == (events_per_sub * 3, 9)
    assert set(df.columns) == timing_cols.union(entity_cols, {'RT', 'PTval'})

    # Long format
    df = run_coll.to_df(format='long')
    assert df.shape == (events_per_sub * 3 * 8, 9)
    assert set(df.columns) == timing_cols.union(entity_cols, condition, ampl)

    # Long format, selecting variables by name
    df = run_coll.to_df(format='long', variables=['RT', 'PTval'])
    assert df.shape == (events_per_sub * 3 * 2, 9)
    assert set(df.columns) == timing_cols.union(entity_cols, condition, ampl)

    # Wide format without entity columnns
    df = run_coll.to_df(format='wide', entities=False)
    assert df.shape == (events_per_sub * 3, 10)
    assert set(df.columns) == timing_cols | cond_names

    # Wide format without timing columns
    df = run_coll.to_df(format='wide', timing=False)
    assert df.shape == (events_per_sub * 3, 13)
    assert set(df.columns) == entity_cols | cond_names


def test_run_variable_collection_to_df_all_dense_vars(run_coll):

    timing_cols = {'onset', 'duration'}
    entity_cols = {'subject', 'run', 'task',  'suffix', 'datatype'}
    cond_names = {'PTval', 'RT', 'gain', 'loss', 'parametric gain', 'respcat',
                  'respnum', 'trial_type'}
    md_names = {'TaskName', 'RepetitionTime', 'extension', 'SliceTiming'}
    condition = {'condition'}
    ampl = {'amplitude'}

    # First few tests assume uniform sampling rate for all columns
    unif_coll = run_coll.to_dense(sampling_rate=10)

    # Fails because all vars are dense
    with pytest.raises(ValueError):
        unif_coll.to_df(include_dense=False)

    # Fails because no such variables exist
    with pytest.raises(ValueError):
        df = unif_coll.to_df(variables=['rubadubdub', '999'])

    # Wide format
    df = unif_coll.to_df()
    rows_per_var = 3 * 3 * 480 * 10  # subjects x runs x time x sampling rate
    assert df.shape == (rows_per_var, 18)
    cols = (timing_cols | entity_cols | cond_names | md_names) - {'trial_type'}
    assert set(df.columns) == cols

    # Wide format with variable selection
    df = unif_coll.to_df(variables=['RT', 'gain'])
    assert df.shape == (rows_per_var, 13)
    cols = (timing_cols | entity_cols | {'RT', 'gain'} | md_names) - {'trial_type'}
    assert set(df.columns) == cols

    # Long format
    df = unif_coll.to_df(format='long')
    assert df.shape == (rows_per_var * 7, 13)
    cols = timing_cols | entity_cols | condition | ampl | md_names
    assert set(df.columns) == cols

    # Long format with variable selection
    df = unif_coll.to_df(format='long', variables=['RT', 'gain'])
    assert df.shape == (rows_per_var * 2, 13)
    cols = timing_cols | entity_cols | condition | ampl | md_names
    assert set(df.columns) == cols

    # Test resampling to TR
    df = unif_coll.to_df(sampling_rate='TR')
    n_rows = int(480 * 3 * 3 / 2)
    assert df.shape == (n_rows, 18)
    cols = (timing_cols | entity_cols | cond_names | md_names) - {'trial_type'}
    assert set(df.columns) == cols

    # Test resampling to highest when variable sampling rates diverge
    unif_coll.resample(variables='RT', sampling_rate=12, in_place=True)
    df = unif_coll.to_df(sampling_rate='highest')
    n_rows = int(rows_per_var * 12 / 10)
    assert df.shape == (n_rows, 18)


def test_run_variable_collection_to_df_mixed_vars(run_coll):
    coll = run_coll.clone()
    coll.to_dense(10, variables=['RT', 'loss','gain'], in_place=True)

    # Only sparse variables
    df = coll.to_df(include_dense=False)
    rows = 3 * 256
    assert df.shape == (rows, 12)
    assert 'RT' not in df.columns
    assert 'respcat' in df.columns

    # Only dense variables
    df = coll.to_df(include_sparse=False)
    rows = 3 * 3 * 480 * 10
    assert df.shape == (rows, 14)
    assert 'RT' in df.columns
    assert 'respcat' not in df.columns

    # Everything
    df = coll.to_df(sampling_rate=5)
    rows = 3 * 3 * 480 * 5
    assert df.shape == (rows, 18)
    assert not {'RT', 'respcat'} - set(df.columns)


def test_merge_collections(run_coll, run_coll_list):
    df1 = run_coll.to_df().sort_values(['subject', 'run', 'onset'])
    rcl = [c.clone() for c in run_coll_list]
    coll = merge_collections(rcl)
    df2 = coll.to_df().sort_values(['subject', 'run', 'onset'])
    assert df1.equals(df2)


def test_get_collection_entities(run_coll_list):
    coll = run_coll_list[0]
    ents = coll.entities
    assert not ({'run', 'task', 'subject', 'suffix', 'datatype'} - set(ents.keys()))

    merged = merge_collections(run_coll_list[:3])
    ents = merged.entities
    assert not ({'task', 'subject', 'suffix', 'datatype'} - set(ents.keys()))
    assert ents['subject'] == '01'

    merged = merge_collections(run_coll_list[3:6])
    ents = merged.entities
    assert not ({'task', 'subject', 'suffix', 'datatype'} - set(ents.keys()))
    assert ents['subject'] == '02'


def test_match_variables(run_coll):
    matches = run_coll.match_variables('^.{1,2}a', match_type='regex')
    assert set(matches) == {'gain', 'parametric gain'}
    assert not run_coll.match_variables('.{1,3}a')
    matches = run_coll.match_variables('^.{1,2}a', match_type='regex',
                                       return_type='variable')
    assert len(matches) == 2
    assert all([isinstance(m, SparseRunVariable) for m in matches])
    matches = run_coll.match_variables('*gain')
    assert set(matches) == {'gain', 'parametric gain'}
