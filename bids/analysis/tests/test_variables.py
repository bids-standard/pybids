from bids.grabbids import BIDSLayout
# from bids.analysis.variables.base import Run, Session, Subject, Dataset
import pytest
from os.path import join, dirname, abspath
from bids import grabbids
from bids.analysis import load_variables
from bids.analysis.variables import merge_variables, DenseRunVariable
from bids.analysis.variables.entities import RunInfo
import numpy as np
# # from grabbit import merge_layouts
# import tempfile
# import shutil


@pytest.fixture(scope="module")
def layout1():
    mod_file = abspath(grabbids.__file__)
    path = join(dirname(mod_file), 'tests', 'data', 'ds005')
    layout = BIDSLayout(path)
    return layout


@pytest.fixture(scope="module")
def layout2():
    mod_file = abspath(grabbids.__file__)
    path = join(dirname(mod_file), 'tests', 'data', '7t_trt')
    layout = BIDSLayout(path)
    return layout


def test_dense_event_variable():
    sr, duration = 20, 480
    n = duration * sr
    values = np.random.normal(size=n)
    entities = {'subject': '01', 'run': 1, 'session': 1, 'task': 'sit'}
    run_info = RunInfo(1, entities, duration, 2)
    dev = DenseRunVariable('test', values, run_info, sr)
    assert len(dev.values) == len(dev.entities)
    dev2 = dev.clone().resample(sampling_rate=40)
    assert len(dev2.values) == len(dev2.entities)
    assert len(dev2.values) == 2 * len(dev.values)


def test_merge_simple_variables(layout2):
    dataset = load_variables(layout2, 'sessions')
    variables = [s.variables['panas_sad'] for s in dataset.children.values()]
    n_rows = sum([len(c.values) for c in variables])
    merged = merge_variables(variables)
    assert len(merged.values) == n_rows
    assert merged.entities.columns.equals(variables[0].entities.columns)
    assert variables[3].values.iloc[1] == merged.values.iloc[7]


def test_merge_sparse_event_variables(layout1):
    dataset = load_variables(layout1, 'events', scan_length=480)
    runs = dataset.get_runs()
    variables = [r.variables['RT'] for r in runs]
    n_rows = sum([len(c.values) for c in variables])
    merged = merge_variables(variables)
    assert len(merged.values) == n_rows
    assert merged.entities.columns.equals(variables[0].entities.columns)
