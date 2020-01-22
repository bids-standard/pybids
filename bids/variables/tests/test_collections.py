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
    return layout.get_collections('run', types=['events'], merge=True,
                                  scan_length=480)


@pytest.fixture(scope="module")
def run_coll_list():
    path = join(get_test_data_path(), 'ds005')
    layout = BIDSLayout(path)
    return layout.get_collections('run', types=['events'], merge=False,
                                  scan_length=480)


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
    assert coll._get_sampling_rate('highest') is None
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
    targ_len = 480 * 16 * 3 * 10
    assert all([len(v.values) == targ_len for v in resampled.values()])

    sr = 20
    resampled = run_coll.resample(sr, force_dense=True).variables
    targ_len = 480 * 16 * 3 * sr
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

    # Wide format
    df = run_coll.to_df()
    assert df.shape == (4096, 15) 
    assert set(df.columns) == timing_cols.union(entity_cols, cond_names)

    # Wide format, selecting variables by name
    df = run_coll.to_df(format='wide', variables=['RT', 'PTval'])
    assert df.shape == (4096, 9)
    assert set(df.columns) == timing_cols.union(entity_cols, {'RT', 'PTval'})

    # Long format
    df = run_coll.to_df(format='long')
    assert df.shape == (4096 * 8, 9)
    assert set(df.columns) == timing_cols.union(entity_cols, condition, ampl)

    # Long format, selecting variables by name
    df = run_coll.to_df(format='long', variables=['RT', 'PTval'])
    assert df.shape == (4096 * 2, 9)
    assert set(df.columns) == timing_cols.union(entity_cols, condition, ampl)

    # Wide format without entity columnns
    df = run_coll.to_df(format='wide', entities=False)
    assert df.shape == (4096, 10)
    assert set(df.columns) == timing_cols | cond_names

    # Wide format without timing columns
    df = run_coll.to_df(format='wide', timing=False)
    assert df.shape == (4096, 13)
    assert set(df.columns) == entity_cols | cond_names


def test_run_variable_collection_to_df_all_dense_vars(run_coll):
    run_coll = run_coll.clone()
    # All variables dense, wide format
    df = run_coll.to_df(sparse=False)
    assert df.shape == (230400, 18)
    extra_cols = {'TaskName', 'RepetitionTime', 'extension', 'SliceTiming'}
    assert set(df.columns) == (wide_cols | extra_cols) - {'trial_type'}

    # All variables dense, wide format
    df = run_coll.to_df(sparse=False, format='long')
    assert df.shape == (1612800, 13)
    assert set(df.columns) == (long_cols | extra_cols)


def test_run_variable_collection_to_df_mixed_vars(run_coll):
    pass

def test_run(analysis):
    kwargs = dict(run=1, subject='01')
    collections = analysis['run'].get_collections(**kwargs)
    assert len(collections) == 1
    sparse_coll = collections[0]

    # Long format, all variables sparse
    result = sparse_coll.to_df(format='long')
    assert result.shape == (172, 9)

    # Long format, include only dense, but there are none, so it fails
    result = collections[0].to_df(include_sparse=False, format='long')
    assert result is None

    # Check that mixed collections are handled properly

    result = collections[0].to_df(sparse=False, include_sparse=True,
                                  sampling_rate='highest', format='wide',
                                  timing=False)
    print(result.columns)
    assert result.shape == (4800, 10)

    result = collections[0].get_design_matrix(mode='dense', force=True,
                                        sampling_rate='TR')
    assert result.shape == (240, 10)

    result = collections[0].get_design_matrix(mode='dense', force=True,
                                        sampling_rate=0.5)
    assert result.shape == (240, 10)

    # format='long' should be ignored for dense output
    result = collections[0].get_design_matrix(mode='dense', force=True, format='long',
                                        entities=False)
    assert result.shape == (240, 1)

    result = collections[0].get_design_matrix(mode='sparse', format='wide',
                                        entities=False)
    assert result.shape == (86, 4)


def test_merge_collections(run_coll, run_coll_list):
    df1 = run_coll.to_df().sort_values(['subject', 'run', 'onset'])
    rcl = [c.clone() for c in run_coll_list]
    coll = merge_collections(rcl)
    df2 = coll.to_df().sort_values(['subject', 'run', 'onset'])
    assert df1.equals(df2)


def test_get_collection_entities(run_coll_list):
    coll = run_coll_list[0]
    ents = coll.entities
    assert {'run', 'task', 'subject', 'suffix', 'datatype'} == set(ents.keys())

    merged = merge_collections(run_coll_list[:3])
    ents = merged.entities
    assert {'task', 'subject', 'suffix', 'datatype'} == set(ents.keys())
    assert ents['subject'] == '01'

    merged = merge_collections(run_coll_list[3:6])
    ents = merged.entities
    assert {'task', 'subject', 'suffix', 'datatype'} == set(ents.keys())
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
