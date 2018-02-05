from bids.grabbids import BIDSLayout
from bids.variables import (SparseRunVariable, SimpleVariable, load_variables)
from bids.variables.entities import Run, Dataset
import pytest
from os.path import join, dirname, abspath
from bids import grabbids


@pytest.fixture
def layout1():
    mod_file = abspath(grabbids.__file__)
    path = join(dirname(mod_file), 'tests', 'data', 'ds005')
    return BIDSLayout(path)


@pytest.fixture(scope="module")
def layout2():
    mod_file = abspath(grabbids.__file__)
    path = join(dirname(mod_file), 'tests', 'data', '7t_trt')
    layout = BIDSLayout(path)
    return layout


def test_load_events(layout1):
    dataset = load_variables(layout1, types='events', scan_length=480)
    runs = dataset.get_nodes(level='run', subject='01')
    assert len(runs) == 3
    assert isinstance(runs[0], Run)
    variables = runs[0].variables
    assert len(variables) == 8
    targ_cols = {'parametric gain', 'PTval', 'trial_type', 'respnum'}
    assert not (targ_cols - set(variables.keys()))
    assert isinstance(variables['parametric gain'], SparseRunVariable)
    assert variables['parametric gain'].entities.shape == (86, 4)
    assert variables['parametric gain'].source == 'events'


def test_load_participants(layout1):
    dataset = load_variables(layout1, types='participants')
    assert isinstance(dataset, Dataset)
    assert len(dataset.variables) == 2
    assert {'age', 'sex'} == set(dataset.variables.keys())
    age = dataset.variables['age']
    assert isinstance(age, SimpleVariable)
    assert age.entities.shape == (16, 1)
    assert age.values.shape == (16,)
