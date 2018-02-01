from bids.grabbids import BIDSLayout
from bids.analysis.variables.base import Run, Session, Subject, Dataset
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
    run = Run(1, None, img.filename, 480, 2)
    assert run.image_file == img.filename
    assert run.duration == 480
    assert run.repetition_time == 2
    assert run.parent is None


def test_get_or_create_node(layout1):
    img = layout1.get(subject='01', task='mixedgamblestask', type='bold',
                      run=1, return_type='obj')[0]
    dataset = Dataset()

    entities = {'subject': '01', 'session': 1}
    sess = dataset.get_or_create_node(entities)
    assert sess.__class__ == Session
    assert sess.children == {}
    assert len(dataset.children) == 1

    sess2 = dataset.get_or_create_node(entities)
    assert sess2 == sess

    run = dataset.get_or_create_node(img.entities, image_file=img.filename,
                                     duration=480, repetition_time=2)
    assert run.__class__ == Run
    assert run.parent == sess
    assert run.duration == 480
