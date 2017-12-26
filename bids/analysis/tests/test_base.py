from os.path import join, dirname, abspath
from bids import grabbids
from bids.grabbids import BIDSLayout
from bids.analysis.base import Analysis, Block
from bids.analysis.variables import load_variables, load_event_variables
import pytest


@pytest.fixture
def analysis():
    mod_file = abspath(grabbids.__file__)
    layout_path = join(dirname(mod_file), 'tests', 'data', 'ds005')
    layout = BIDSLayout(layout_path)
    json_file = join(layout_path, 'models', 'ds-005_type-test_model.json')

    # Load variables manually because we need to specify the scan length
    variables = load_variables(layout, levels=['run', 'session', 'subject'])
    variables['time'] = load_event_variables(layout, scan_length=480)

    analysis = Analysis(layout_path, json_file, variables=variables)
    analysis.setup()
    return analysis


def test_analysis_smoke_test(analysis):

    result = analysis['run'].get_Xy(subject=['01', '02'])
    assert len(result) == 6
    assert len(result[0]) == 2

    result = analysis['session'].get_Xy(drop_entities=True)
    assert len(result) == 16
    assert len(result[0]) == 2
    assert result[0].data.shape == (24, 2)
    assert result[0].entities == {'subject': 1}

    # Participant level and also check integer-based indexing
    result = analysis['participant'].get_Xy()
    assert len(result) == 16
    assert analysis[2].name == 'participant'

    # Dataset level
    result = analysis['group'].get_Xy()
    assert len(result) == 1
    data = result[0].data
    assert len(data) == 160
    # Not 16 because subs get represented as both ints and str--should fix!
    assert data['subject'].nunique() == 32
    # Make sure columns from different levels exist
    varset = {'sex', 'age', 'RT', 'respnum'}
    assert not (varset - set(data['condition'].unique()))

    # Calling an invalid level name should raise an exception
    with pytest.raises(KeyError):
        result = analysis['nonexistent_name'].get_Xy()

    # With entities included
    result = analysis['run'].get_Xy(drop_entities=False)
    assert len(result) == 48
    assert result[0][0].shape == (688, 9)
    assert set(result[0][0].columns) == {'amplitude', 'onset', 'duration',
                                         'condition', 'subject', 'run',
                                         'task', 'modality', 'type'}
