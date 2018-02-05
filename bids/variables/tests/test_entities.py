from bids.grabbids import BIDSLayout
from bids.variables.entities import Run, Session, Dataset
from bids.variables import load_variables
from bids.variables import BIDSRunVariableCollection
import pytest
from os.path import join, dirname, abspath
from bids import grabbids


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


def test_run(layout1):
    img = layout1.get(subject='01', task='mixedgamblestask', type='bold',
                      run=1, return_type='obj')[0]
    run = Run(1, None, img.filename, 480, 2, 'mixedgamblestask')
    assert run.image_file == img.filename
    assert run.duration == 480
    assert run.repetition_time == 2
    assert run.parent is None


def test_get_or_create_node(layout1):
    img = layout1.get(subject='01', task='mixedgamblestask', type='bold',
                      run=1, return_type='obj')[0]
    dataset = Dataset()

    entities = {'subject': '01', 'session': 1, 'task': 'test'}
    sess = dataset.get_or_create_node(entities)
    assert sess.__class__ == Session
    assert sess.children == {}
    assert len(dataset.children) == 1

    sess2 = dataset.get_or_create_node(entities)
    assert sess2 == sess

    run = dataset.get_or_create_node(img.entities, image_file=img.filename,
                                     duration=480, repetition_time=2,
                                     task='mixedgamblestask')
    assert run.__class__ == Run
    assert run.parent == sess
    assert run.duration == 480


def test_get_nodes(layout1):
    dataset = load_variables(layout1, scan_length=480)
    nodes = dataset.get_nodes('session')
    assert len(nodes) == 16
    assert all([isinstance(n, Session) for n in nodes])
    nodes = dataset.get_nodes('run', subject=['01', '02', '03'])
    assert len(nodes) == 9
    assert all([isinstance(n, Run) for n in nodes])


def test_get_collections_merged(layout1):
    dataset = load_variables(layout1, scan_length=480)
    collection = dataset.get_collections('run', merge=True)
    assert isinstance(collection, BIDSRunVariableCollection)
    assert len(collection.variables) == 8
    vals = collection.variables['RT'].values
    ents = collection.variables['RT'].entities
    assert len(ents) == len(vals) == 4096
    assert set(ents.columns) == {'task', 'run', 'session', 'subject'}


def test_get_collections_unmerged(layout2):
    dataset = load_variables(layout2, types=['sessions'], scan_length=480)
    colls = dataset.get_collections('subject', merge=False)
    assert len(colls) == 10
    assert len(colls[0].variables) == 94
    assert colls[0]['panas_at_ease'].values.shape == (2,)
