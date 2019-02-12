from bids.layout import BIDSLayout
from bids.variables.entities import RunNode, Node, NodeIndex
from bids.variables import load_variables
from bids.variables import BIDSRunVariableCollection
import pytest
from os.path import join
from bids.tests import get_test_data_path


@pytest.fixture(scope="module")
def layout1():
    path = join(get_test_data_path(), 'ds005')
    layout = BIDSLayout(path)
    return layout


@pytest.fixture(scope="module")
def layout2():
    path = join(get_test_data_path(), '7t_trt')
    layout = BIDSLayout(path)
    return layout


def test_run(layout1):
    img = layout1.get(subject='01', task='mixedgamblestask', suffix='bold',
                      run=1, return_type='obj')[0]
    run = RunNode(None, img.filename, 480, 2)
    assert run.image_file == img.filename
    assert run.duration == 480
    assert run.repetition_time == 2


def test_get_or_create_node(layout1):
    img = layout1.get(subject='01', task='mixedgamblestask', suffix='bold',
                      run=1, return_type='obj')[0]
    index = NodeIndex()

    entities = {'subject': '01', 'session': 1}
    sess = index.get_or_create_node('session', entities)
    assert sess.__class__ == Node

    sess2 = index.get_or_create_node('session', entities)
    assert sess2 == sess

    run = index.get_or_create_node('run', img.entities,
                                   image_file=img.filename, duration=480,
                                   repetition_time=2)
    assert run.__class__ == RunNode
    assert run.duration == 480


def test_get_nodes(layout1):
    index = load_variables(layout1, scan_length=480)
    nodes = index.get_nodes('session')
    assert len(nodes) == 0
    nodes = index.get_nodes('dataset')
    assert len(nodes) == 1
    assert all([isinstance(n, Node) for n in nodes])
    nodes = index.get_nodes('run', {'subject': ['01', '02', '03']})
    assert len(nodes) == 9
    assert all([isinstance(n, RunNode) for n in nodes])


def test_get_collections_merged(layout1):
    dataset = load_variables(layout1, scan_length=480)
    collection = dataset.get_collections('run', merge=True)
    assert isinstance(collection, BIDSRunVariableCollection)
    assert len(collection.variables) == 8
    vals = collection.variables['RT'].values
    ents = collection.variables['RT'].index
    assert len(ents) == len(vals) == 4096
    assert set(ents.columns) == {'task', 'run', 'subject', 'suffix', 'datatype'}


def test_get_collections_unmerged(layout2):
    dataset = load_variables(layout2, types=['sessions'], scan_length=480)
    colls = dataset.get_collections('subject', merge=False)
    assert len(colls) == 10
    assert len(colls[0].variables) == 94
    assert colls[0]['panas_at_ease'].values.shape == (2,)
