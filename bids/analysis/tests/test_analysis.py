from os.path import join
from bids.grabbids import BIDSLayout
from bids.analysis import Analysis
from bids.tests import get_test_data_path
import pytest


@pytest.fixture
def analysis():
    layout_path = join(get_test_data_path(), 'ds005')
    layout = BIDSLayout(layout_path)
    json_file = join(layout_path, 'models', 'ds-005_type-test_model.json')
    analysis = Analysis(layout, json_file)
    analysis.setup(scan_length=480, subject=['01', '02'])
    return analysis


def test_analysis_smoke_test(analysis):
    result = analysis['run'].get_design_matrix(subject=['01', '02'])
    assert len(result) == 6
    assert len(result[0]) == 2

    result = analysis['session'].get_design_matrix()
    assert len(result) == 16
    assert len(result[0]) == 2
    assert result[0].data.shape == (24, 2)
    assert result[0].entities == {'subject': '01'}

    # Participant level and also check integer-based indexing
    result = analysis['participant'].get_design_matrix()
    assert len(result) == 16
    assert analysis[2].name == 'participant'

    # Dataset level
    result = analysis['group'].get_design_matrix()
    assert len(result) == 1
    data = result[0].data
    assert len(data) == 160
    assert data['subject'].nunique() == 16
    # Make sure columns from different levels exist
    varset = {'sex', 'age', 'RT', 'respnum'}
    assert not (varset - set(data['condition'].unique()))

    # Calling an invalid level name should raise an exception
    with pytest.raises(KeyError):
        result = analysis['nonexistent_name'].get_design_matrix()

    # With entities included
    result = analysis['run'].get_design_matrix(drop_entities=False)
    assert len(result) == 48
    assert result[0][0].shape == (688, 9)
    assert set(result[0][0].columns) == {'amplitude', 'onset', 'duration',
                                         'condition', 'subject', 'run',
                                         'task', 'modality', 'type'}
