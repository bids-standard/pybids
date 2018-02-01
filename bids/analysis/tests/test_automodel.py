from os.path import join, dirname, abspath
from bids import grabbids
from bids.grabbids import BIDSLayout
from bids.analysis.variables import load_variables, load_event_variables
from bids.analysis.auto_model import auto_model
from bids.analysis.base import Analysis
import pytest


@pytest.fixture
def model():
    mod_file = abspath(grabbids.__file__)
    layout_path = join(dirname(mod_file), 'tests', 'data', 'ds005')
    layout = BIDSLayout(layout_path)
    
    models = auto_model(layout, scan_length=480)
    
    return models[0]

def test_auto_model_analysis(model):
    
    mod_file = abspath(grabbids.__file__)
    layout_path = join(dirname(mod_file), 'tests', 'data', 'ds005')
    layout = BIDSLayout(layout_path)
    
    # Test to make sure an analaysis can be setup from the generated model
    variables = load_variables(layout, levels=['run', 'session', 'subject'])
    variables['time'] = load_event_variables(layout, scan_length=480)
    analysis = Analysis(layout, model, variables=variables)
    analysis.setup()
    
    assert model['name'] == 'ds005_mixedgamblestask'
    
    # run level
    block = model['blocks'][0]
    assert block['name'] == 'run'
    assert block['level'] == 'run'
    assert block['transformations'][0]['name'] == 'factor'
    assert block['model']['HRF_variables'][0] =='trial_type.parametric gain'
    assert block['contrasts'][0]['name'] == 'parametric gain'
    assert block['contrasts'][0]['weights'] == [1]
    
    # subject level
    block = model['blocks'][1]
    assert block['name'] == 'subject'
    assert block['level'] == 'subject'
    assert block['model']['variables'][0] == 'parametric gain'
    assert block['contrasts'][0]['name'] == 'subject_parametric gain'
    
    # dataset level
    block = model['blocks'][2]
    assert block['name'] == 'dataset'
    assert block['level'] == 'dataset'
    assert block['model']['variables'][0] == 'subject_parametric gain'
    assert block['contrasts'][0]['name'] == 'dataset_subject_parametric gain'