from bids.grabbids import BIDSLayout
import pytest
from os.path import join
from bids.tests import get_test_data_path
from bids.analysis import load_variables
from bids.variables import (merge_variables, DenseRunVariable, SimpleVariable)
from bids.variables.entities import RunInfo
import numpy as np
import pandas as pd
import uuid


def generate_DEV(name='test', sr=20, duration=480):
    n = duration * sr
    values = np.random.normal(size=n)
    ent_names = ['task', 'run', 'session', 'subject']
    entities = {e: uuid.uuid4().hex for e in ent_names}
    image = uuid.uuid4().hex + '.nii.gz'
    run_info = RunInfo(1, entities, duration, 2, image)
    return DenseRunVariable('test', values, run_info, 'dummy', sr)


@pytest.fixture
def layout1():
    path = join(get_test_data_path(), 'ds005')
    layout = BIDSLayout(path)
    return layout


@pytest.fixture(scope="module")
def layout2():
    path = join(get_test_data_path(), '7t_trt')
    layout = BIDSLayout(path)
    return layout


def test_dense_event_variable_init():
    dev = generate_DEV()
    assert dev.sampling_rate == 20
    assert dev.run_info[0].duration == 480
    assert dev.source == 'dummy'
    assert len(dev.values) == len(dev.index)


def test_dense_event_variable_resample():
    dev = generate_DEV()
    dev2 = dev.clone().resample(sampling_rate=40)
    assert len(dev2.values) == len(dev2.index)
    assert len(dev2.values) == 2 * len(dev.values)


def test_merge_wrapper():
    dev = generate_DEV()
    data = pd.DataFrame({'amplitude': [4, 3, 2, 5]})
    sev = SimpleVariable('simple', data, 'dummy')
    # Should break if asked to merge different classes
    with pytest.raises(ValueError) as e:
        merge_variables([dev, sev])
    assert "Variables of different classes" in str(e)


def test_sparse_run_variable_to_dense(layout1):
    dataset = load_variables(layout1, types='events', scan_length=480)
    runs = dataset.get_nodes('run', subject=['01', '02'])

    for i, run in enumerate(runs):
        var = run.variables['RT']
        dense = var.to_dense(20)

        # Check that all unique values are identical
        sparse_vals = set(np.unique(var.values.values)) | {0}
        dense_vals = set(np.unique(dense.values.values))
        assert sparse_vals == dense_vals

        assert len(dense.values) > len(var.values)
        assert isinstance(dense, DenseRunVariable)
        assert dense.values.shape == (9600, 1)
        assert len(dense.run_info) == len(var.run_info)
        assert dense.source == 'events'


def test_merge_densified_variables(layout1):
    SR = 10
    dataset = load_variables(layout1, types='events', scan_length=480)
    runs = dataset.get_nodes('run')
    vars_ = [r.variables['RT'].to_dense(SR) for r in runs]
    dense = merge_variables(vars_)
    assert isinstance(dense, DenseRunVariable)
    n_rows = 480 * SR
    assert dense.values.shape == (len(runs) * n_rows, 1)
    for i in range(len(runs)):
        onset = i * n_rows
        offset = onset + n_rows
        run_vals = vars_[i].values
        dense_vals = dense.values.iloc[onset:offset].reset_index(drop=True)
        assert dense_vals.equals(run_vals)


def test_densify_merged_variables(layout1):
    SR = 10
    dataset = load_variables(layout1, types='events', scan_length=480)
    runs = dataset.get_nodes('run')
    vars_ = [r.variables['RT'] for r in runs]
    var = merge_variables(vars_)
    dense = var.to_dense(SR)
    assert isinstance(dense, DenseRunVariable)
    n_rows = 480 * SR
    assert dense.values.shape == (len(runs) * n_rows, 1)
    for i in range(len(runs)):
        onset = i * n_rows
        offset = onset + n_rows
        run_vals = vars_[i].to_dense(SR).values
        dense_vals = dense.values.iloc[onset:offset].reset_index(drop=True)
        assert dense_vals.equals(run_vals)


def test_merge_simple_variables(layout2):
    dataset = load_variables(layout2, types='sessions')
    variables = [s.variables['panas_sad'] for s in dataset.children.values()]
    n_rows = sum([len(c.values) for c in variables])
    merged = merge_variables(variables)
    assert len(merged.values) == n_rows
    assert merged.index.columns.equals(variables[0].index.columns)
    assert variables[3].values.iloc[1] == merged.values.iloc[7]


def test_merge_sparse_run_variables(layout1):
    dataset = load_variables(layout1, types='events', scan_length=480)
    runs = dataset.get_nodes('run')
    variables = [r.variables['RT'] for r in runs]
    n_rows = sum([len(c.values) for c in variables])
    merged = merge_variables(variables)
    assert len(merged.values) == n_rows
    assert merged.index.columns.equals(variables[0].index.columns)


def test_merge_dense_run_variables(layout2):
    variables = [generate_DEV() for i in range(20)]
    variables += [generate_DEV(duration=400) for i in range(8)]
    n_rows = sum([len(c.values) for c in variables])
    merged = merge_variables(variables)
    assert len(merged.values) == n_rows
    assert merged.index.columns.equals(variables[0].index.columns)


def test_simple_variable_to_df(layout1):
    pass


def test_sparse_run_variable_to_df(layout1):
    pass


def test_dense_run_variable_to_df(layout1):
    pass
