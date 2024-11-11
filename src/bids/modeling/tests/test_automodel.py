from os.path import join
from itertools import chain
from bids.modeling.auto_model import auto_model
from bids.modeling import BIDSStatsModelsGraph
from bids.layout import BIDSLayout
from bids.tests import get_test_data_path
import pytest


@pytest.fixture
def model():
    layout_path = join(get_test_data_path(), 'ds005')
    layout = BIDSLayout(layout_path)

    models = auto_model(layout, scan_length=480, one_vs_rest=True)

    return models[0]

def test_automodel_valid(model):
    try:
        from bsmschema.models import BIDSStatsModel
    except ImportError:
        pytest.skip("Needs bsmschema, available for Python 3.8+")
    BIDSStatsModel.parse_obj(model)

def test_automodel_runs(model):
    layout_path = join(get_test_data_path(), 'ds005')
    layout = BIDSLayout(layout_path)

    # Test to make sure an analaysis can be setup from the generated model
    graph = BIDSStatsModelsGraph(layout, model)
    graph.load_collections(scan_length=480, subject=["01", "02"])
    outputs = graph["Run"].run()
    assert len(outputs) == 6
    cis = list(chain(*[op.contrasts for op in outputs]))
    assert len(cis) == 6
    outputs = graph["Subject"].run(cis)
    # 2 subjects x 1 contrast
    assert len(outputs) == 2
    cis = list(chain(*[op.contrasts for op in outputs]))
    assert len(cis) == 2

def test_auto_model_graph(model):

    assert model['Name'] == 'ds005_mixedgamblestask'

    # run level
    block = model['Nodes'][0]
    assert block['Name'] == 'Run'
    assert block['Level'] == 'Run'
    assert block['Model']['Type'] == 'glm'
    assert block['Transformations']['Transformer'] == 'pybids-transforms-v1'
    assert block['Transformations']['Instructions'][0]['Name'] == 'Factor'
    assert block['Contrasts'][0]['Name'] == 'run_parametric gain'
    assert block['Contrasts'][0]['Weights'] == [1]
    assert block['Contrasts'][0]['Test'] == 't'

    # subject level
    block = model['Nodes'][1]
    assert block['Name'] == 'Subject'
    assert block['Level'] == 'Subject'
    assert block['Model']['Type'] == 'meta'
    assert block['Model']['X'][0] == 1
    assert block['DummyContrasts'] == {'Test': 't'}

    # dataset level
    block = model['Nodes'][2]
    assert block['Name'] == 'Dataset'
    assert block['Level'] == 'Dataset'
    assert block['Model']['Type'] == 'glm'
    assert block['Model']['X'][0] == 1
    assert block['DummyContrasts'] == {'Test': 't'}
