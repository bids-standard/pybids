from bids.layout import BIDSLayout
import pytest
import os
from os.path import join
from bids.tests import get_test_data_path
from bids.variables import (merge_variables, DenseRunVariable, SimpleVariable,
                            load_variables)
from bids.variables.entities import RunInfo
import numpy as np
import pandas as pd
import nibabel as nb
import uuid
import json


def generate_DEV(name='test', sr=20, duration=480):
    n = duration * sr
    values = np.random.normal(size=n)
    ent_names = ['task', 'run', 'session', 'subject']
    entities = {e: uuid.uuid4().hex for e in ent_names}
    image = uuid.uuid4().hex + '.nii.gz'
    run_info = RunInfo(entities, duration, 2, image)
    return DenseRunVariable(name='test', values=values, run_info=run_info,
                            source='dummy', sampling_rate=sr)


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
    with pytest.raises(ValueError, match='Variables of different classes'):
        merge_variables([dev, sev])


def test_sparse_run_variable_to_dense(layout1):
    index = load_variables(layout1, types='events', scan_length=480)
    runs = index.get_nodes('run', {'subject': ['01', '02']})

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


def test_sparse_run_variable_to_dense_default_sr(layout1):
    index = load_variables(layout1, types='events', scan_length=480)
    runs = index.get_nodes('run', {'subject': ['01', '02']})

    for i, run in enumerate(runs):
        var = run.variables['RT']
        dense = var.to_dense()

        # Check that a sensible sampling rate was found
        assert np.allclose(dense.sampling_rate, 1)

        # Check that all unique values are identical
        sparse_vals = set(np.unique(var.values.values)) | {0}
        dense_vals = set(np.unique(dense.values.values))
        assert sparse_vals == dense_vals

        assert len(dense.values) > len(var.values)
        assert isinstance(dense, DenseRunVariable)
        assert dense.values.shape == (480, 1)
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
    index = load_variables(layout2, types='sessions')
    subjects = index.get_nodes('subject')
    variables = [s.variables['panas_sad'] for s in subjects]
    n_rows = sum([len(c.values) for c in variables])
    merged = merge_variables(variables)
    assert len(merged.values) == n_rows
    assert set(merged.index.columns) == set(variables[0].index.columns)
    assert variables[3].values.iloc[1] == merged.values.iloc[7]


def test_merge_sparse_run_variables(layout1):
    dataset = load_variables(layout1, types='events', scan_length=480)
    runs = dataset.get_nodes('run')
    variables = [r.variables['RT'] for r in runs]
    n_rows = sum([len(c.values) for c in variables])
    merged = merge_variables(variables)
    assert len(merged.values) == n_rows
    assert set(merged.index.columns) == set(variables[0].index.columns)


def test_merge_dense_run_variables(layout2):
    variables = [generate_DEV() for i in range(20)]
    variables += [generate_DEV(duration=400) for i in range(8)]
    n_rows = sum([len(c.values) for c in variables])
    merged = merge_variables(variables)
    assert len(merged.values) == n_rows
    assert set(merged.index.columns) == set(variables[0].index.columns)


def test_simple_variable_to_df(layout1):
    pass


def test_sparse_run_variable_to_df(layout1):
    pass


def test_dense_run_variable_to_df(layout2):
    pass


def test_filter_simple_variable(layout2):
    dataset = load_variables(layout2, types=['scans'])
    sessions = dataset.get_nodes('session')
    variables = [s.variables['surroundings'] for s in sessions]
    merged = merge_variables(variables)
    assert merged.to_df().shape == (60, 9)
    filt = merged.filter({'acquisition': 'fullbrain'})
    assert filt.to_df().shape == (40, 9)
    flt1 = merged.filter({'acquisition': 'fullbrain',
                          'subject': ['01', '02']}).to_df()
    assert flt1.shape == (8, 9)
    query = 'acquisition=="fullbrain" and subject in ["01", "02"]'
    flt2 = merged.filter(query=query)
    flt2 = flt2.to_df()
    assert flt1.equals(flt2)
    assert merged.filter({'nonexistent': 2}, strict=True) is None
    merged.filter({'acquisition': 'fullbrain'}, inplace=True)
    assert merged.to_df().shape == (40, 9)


@pytest.mark.parametrize(
    "TR, nvols",
    [(2.00000, 251),
     (2.000001, 251)])
def test_resampling_edge_case(tmpdir, TR, nvols):
    tmpdir.chdir()
    os.makedirs('sub-01/func')
    with open('sub-01/func/sub-01_task-task_events.tsv', 'w') as fobj:
        fobj.write('onset\tduration\tval\n1\t0.1\t1\n')
    with open('sub-01/func/sub-01_task-task_bold.json', 'w') as fobj:
        json.dump({'RepetitionTime': TR}, fobj)

    dataobj = np.zeros((5, 5, 5, nvols), dtype=np.int16)
    affine = np.diag((2.5, 2.5, 2.5, 1))
    img = nb.Nifti1Image(dataobj, affine)
    img.header.set_zooms((2.5, 2.5, 2.5, TR))
    img.to_filename('sub-01/func/sub-01_task-task_bold.nii.gz')

    layout = BIDSLayout('.', validate=False)
    coll = load_variables(layout).get_collections('run')[0]
    dense_var = coll.variables['val'].to_dense(coll.sampling_rate)
    regressor = dense_var.resample(1.0 / TR).values
    assert regressor.shape == (nvols, 1)


def test_downsampling(tmpdir):
    tmpdir.chdir()
    os.makedirs('sub-01/func')
    import numpy as np
    TR, newTR, nvols, newvols = 2.00000, 6.0, 90, 30
    Fs = 1 / TR
    t = np.linspace(0, int(nvols / Fs), nvols, endpoint=False)
    values = np.sin(0.025 * 2 * np.pi * t) + np.cos(0.1166 * 2 * np.pi * t)
    with open('sub-01/func/sub-01_task-task_events.tsv', 'w') as fobj:
        fobj.write('onset\tduration\tval\n')
        for idx, val in enumerate(values):
            fobj.write('%f\t%f\t%f\n' % (idx*TR, TR, val))
    with open('sub-01/func/sub-01_task-task_bold.json', 'w') as fobj:
        json.dump({'RepetitionTime': TR}, fobj)

    dataobj = np.zeros((5, 5, 5, nvols), dtype=np.int16)
    affine = np.diag((2.5, 2.5, 2.5, 1))
    img = nb.Nifti1Image(dataobj, affine)
    img.header.set_zooms((2.5, 2.5, 2.5, TR))
    img.to_filename('sub-01/func/sub-01_task-task_bold.nii.gz')

    layout = BIDSLayout('.', validate=False)
    coll = load_variables(layout).get_collections('run')[0]
    dense_var = coll.variables['val'].to_dense(1.0 / TR)
    regressor = dense_var.resample(1.0 / newTR).values
    assert regressor.shape == (newvols, 1)
    # This checks that the filtering has happened. If it has not, then
    # this value for this frequency bin will be an alias and have a
    # very different amplitude
    assert np.allclose(np.abs(np.fft.fft(regressor.values.ravel()))[9],
                       0.46298273)
    # This checks that the signal (0.025 Hz) within the new Nyquist
    # rate actually gets passed through.
    assert np.allclose(np.abs(np.fft.fft(regressor.values.ravel()))[4],
                       8.88189504)
