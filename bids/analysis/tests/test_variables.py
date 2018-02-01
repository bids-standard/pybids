from bids.grabbids import BIDSLayout
# from bids.analysis.variables.base import Run, Session, Subject, Dataset
import pytest
from os.path import join, dirname, abspath
from bids import grabbids
from bids.analysis import load_variables
from bids.analysis.variables import merge_variables
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
