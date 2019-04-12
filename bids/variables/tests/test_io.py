from bids.layout import BIDSLayout
from bids.variables import (SparseRunVariable, SimpleVariable,
                            DenseRunVariable, load_variables)
from bids.variables.entities import Node, RunNode, NodeIndex
import pytest
from os.path import join
from bids.tests import get_test_data_path
from bids.config import set_option, get_option


@pytest.fixture
def layout1():
    path = join(get_test_data_path(), 'ds005')
    layout = BIDSLayout(path)
    return layout


@pytest.fixture(scope="module", params=["events", "preproc"])
def synthetic(request):
    root = join(get_test_data_path(), 'synthetic')
    if request.param == 'preproc':
        layout = BIDSLayout(root, derivatives=True)
        dataset = load_variables(layout, skip_empty=True, desc='preproc',
                                 space='T1w')
    else:
        layout = BIDSLayout(root)
        dataset = load_variables(layout, skip_empty=True)
    yield request.param, dataset


def test_load_events(layout1):
    index = load_variables(layout1, types='events', scan_length=480)
    runs = index.get_nodes(level='run', entities={'subject': '01'})
    assert len(runs) == 3
    assert isinstance(runs[0], RunNode)
    variables = runs[0].variables
    assert len(variables) == 8
    targ_cols = {'parametric gain', 'PTval', 'trial_type', 'respnum'}
    assert not (targ_cols - set(variables.keys()))
    assert isinstance(variables['parametric gain'], SparseRunVariable)
    assert variables['parametric gain'].index.shape == (86, 5)
    assert variables['parametric gain'].source == 'events'


def test_load_participants(layout1):
    index = load_variables(layout1, types='participants')
    assert isinstance(index, NodeIndex)
    dataset = index.get_nodes(level='dataset')[0]
    assert isinstance(dataset, Node)
    assert len(dataset.variables) == 2
    assert {'age', 'sex'} == set(dataset.variables.keys())
    age = dataset.variables['age']
    assert isinstance(age, SimpleVariable)
    assert age.index.shape == (16, 2)
    assert age.values.shape == (16,)

    index = load_variables(layout1, types='participants', subject=['^1.*'])
    age = index.get_nodes(level='dataset')[0].variables['age']
    assert age.index.shape == (7, 2)
    assert age.values.shape == (7,)


def test_load_synthetic_dataset(synthetic):

    param, index = synthetic
    # Runs
    runs = index.get_nodes('run')
    assert len(runs) == 5 * 2 * 3
    runs = index.get_nodes('run', {'task': 'nback'})
    assert len(runs) == 5 * 2 * 2
    variables = runs[0].variables

    if param == 'preproc':
        # non-exhaustive
        match = ['Cosine01', 'stdDVARS', 'RotZ', 'FramewiseDisplacement']
        sum_dense = 56
    else:
        match = ['trial_type', 'weight', 'respiratory', 'cardiac', 'stimA',
                 'stimB']
        sum_dense = 4

    for v in match:
        assert v in variables.keys()

    assert sum([isinstance(v, DenseRunVariable)
                for v in variables.values()]) == sum_dense
    assert all([len(r.variables['weight'].values) == 42 for r in runs])

    # Sessions
    sessions = index.get_nodes('session')
    assert len(sessions) == 5 * 2
    target_cols = {'acq_time', 'TaskName', 'RepetitionTime'}
    assert set(sessions[0].variables.keys()) == target_cols
    data = sessions[0].variables['acq_time'].filter({'task': 'nback'})
    assert len(data.values) == 2

    # Subjects
    subs = index.get_nodes('subject')
    assert len(subs) == 5
    assert set(subs[0].variables.keys()) == {'systolic_blood_pressure'}
