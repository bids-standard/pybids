from bids.grabbids import BIDSLayout
import pytest
from os.path import join, dirname, abspath
from bids import grabbids
from bids.variables import DenseRunVariable


@pytest.fixture(scope="module")
def run_coll():
    mod_file = abspath(grabbids.__file__)
    path = join(dirname(mod_file), 'tests', 'data', 'ds005')
    layout = BIDSLayout(path)
    return layout.get_variables('run', ['events'], merge=True, scan_length=480)


def test_run_variable_collection_init(run_coll):
    assert isinstance(run_coll.variables, dict)
    assert run_coll.sampling_rate == 10


def test_resample_run_variable_collection(run_coll):
    run_coll = run_coll.clone()
    resampled = run_coll.resample()
    assert not resampled  # Empty because all variables are sparse

    resampled = run_coll.resample(force_dense=True)
    assert len(resampled) == 7
    assert all([isinstance(v, DenseRunVariable) for v in resampled.values()])
    assert len(set([v.sampling_rate for v in resampled.values()])) == 1
    targ_len = 480 * 16 * 3 * 10
    assert all([len(v.values) == targ_len for v in resampled.values()])

    sr = 20
    resampled = run_coll.resample(sr, force_dense=True)
    targ_len = 480 * 16 * 3 * sr
    assert all([len(v.values) == targ_len for v in resampled.values()])

    run_coll.resample(sr, force_dense=True, in_place=True)
    assert len(run_coll.variables) == 8
    vars_ = run_coll.variables.values()
    vars_ = [v for v in vars_ if v.name != 'trial_type']
    assert all([len(v.values) == targ_len for v in vars_])
    assert all([v.sampling_rate == sr for v in vars_])
    assert all([isinstance(v, DenseRunVariable) for v in vars_])


def test_run_variable_collection_to_df(run_coll):
    run_coll = run_coll.clone()

    # All variables sparse, wide format
    df = run_coll.to_df()
    assert df.shape == (4096, 14)
    wide_cols = {'onset', 'duration', 'subject', 'run', 'task', 'session',
                 'PTval', 'RT', 'gain', 'loss', 'parametric gain', 'respcat',
                 'respnum', 'trial_type'}
    assert set(df.columns) == wide_cols

    # All variables sparse, wide format
    df = run_coll.to_df(format='long')
    assert df.shape == (32768, 8)
    long_cols = {'amplitude', 'duration', 'onset', 'condition', 'run',
                 'session', 'task', 'subject'}
    assert set(df.columns) == long_cols

    # All variables dense, wide format
    df = run_coll.to_df(sparse=False)
    assert df.shape == (230400, 13)
    assert set(df.columns) == wide_cols - {'trial_type'}

    # All variables dense, wide format
    df = run_coll.to_df(sparse=False, format='long')
    assert df.shape == (1612800, 8)
    assert set(df.columns) == long_cols
