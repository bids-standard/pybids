from os.path import join
from bids.analysis.auto_model import auto_model
from bids.analysis import Analysis
from bids.layout import BIDSLayout
from bids.tests import get_test_data_path
import pytest


@pytest.fixture
def model():
    layout_path = join(get_test_data_path(), 'ds005')
    layout = BIDSLayout(layout_path)

    models = auto_model(layout, scan_length=480, one_vs_rest=True)

    return models[0]


def test_auto_model_analysis(model):

    layout_path = join(get_test_data_path(), 'ds005')
    layout = BIDSLayout(layout_path)

    # Test to make sure an analaysis can be setup from the generated model
    analysis = Analysis(layout, model)
    analysis.setup(scan_length=480)

    assert model['name'] == 'ds005_mixedgamblestask'

    # run level
    block = model['blocks'][0]
    assert block['name'] == 'run'
    assert block['level'] == 'run'
    assert block['transformations'][0]['name'] == 'factor'
    assert block['model']['HRF_variables'][0] == 'trial_type.parametric gain'
    assert block['contrasts'][0]['name'] == 'run_parametric gain'
    assert block['contrasts'][0]['weights'] == [1]

    # subject level
    block = model['blocks'][1]
    assert block['name'] == 'subject'
    assert block['level'] == 'subject'
    assert block['model']['variables'][0] == 'run_parametric gain'
    assert block['contrasts'][0]['name'] == 'subject_run_parametric gain'

    # dataset level
    block = model['blocks'][2]
    assert block['name'] == 'dataset'
    assert block['level'] == 'dataset'
    assert block['model']['variables'][0] == 'subject_run_parametric gain'
    assert block['contrasts'][0]['name'] == 'dataset_subject_run_parametric gain'
