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


def test_design_matrix_info(analysis):
    result = analysis['run'].get_design_matrix(subject=['01', '02', '03'])
    for dmi in result:
        assert dmi._fields == ('sparse', 'dense', 'entities')
        assert hasattr(dmi.sparse, 'shape')
        assert dmi.dense is None
        assert isinstance(dmi.entities, dict)


def test_get_design_matrix_arguments(analysis):
    kwargs = dict(run='1', subject='01', sparse=True)
    result = analysis['run'].get_design_matrix(**kwargs)[0]
    assert result.sparse.shape == (688, 8)
    assert result.dense is None

    kwargs = dict(run='1', subject='01', mode='dense', force=False)
    result = analysis['run'].get_design_matrix(**kwargs)[0]
    assert result.sparse is None
    assert result.dense is None

    kwargs = dict(run='1', subject='01', mode='dense', force=True)
    result = analysis['run'].get_design_matrix(**kwargs)[0]
    assert result.sparse is None
    assert result.dense.shape == (4800, 11)

    # format='long' should be ignored for dense output
    kwargs = dict(run='1', subject='01', mode='dense', force=True,
                  format='long', entities=False)
    result = analysis['run'].get_design_matrix(**kwargs)[0]
    assert result.sparse is None
    assert result.dense.shape == (4800, 7)

    kwargs = dict(run='1', subject='01', mode='sparse', format='wide',
                  entities=False)
    result = analysis['run'].get_design_matrix(**kwargs)[0]
    assert result.dense is None
    assert result.sparse.shape == (86, 10)


def test_first_level_sparse_design_matrix(analysis):
    result = analysis['run'].get_design_matrix(subject=['01'])
    assert len(result) == 3
    df = result[0].sparse
    assert df.shape == (688, 8)
    assert df['condition'].nunique() == 8
    assert set(result[0][0].columns) == {'amplitude', 'onset', 'duration',
                                         'condition', 'subject', 'run',
                                         'task', 'session'}


def test_post_first_level_sparse_design_matrix(analysis):

    result = analysis['session'].get_design_matrix(entities=False)
    assert len(result) == 2
    assert len(result[0]) == 3
    assert result[0].sparse.shape == (24, 2)
    assert result[0].entities == {
        'session': 1,
        'subject': '01',
        'task': 'mixedgamblestask'}

    # Participant level and also check integer-based indexing
    result = analysis['participant'].get_design_matrix()
    assert len(result) == 2
    assert analysis[2].name == 'participant'

    # Dataset level
    result = analysis['group'].get_design_matrix()
    assert len(result) == 1
    data = result[0].sparse
    assert len(data) == 20
    assert data['subject'].nunique() == 2

    # # Make sure columns from different levels exist
    varset = {'sex', 'age', 'RT', 'respnum'}
    assert not (varset - set(data['condition'].unique()))

    # Calling an invalid level name should raise an exception
    with pytest.raises(KeyError):
        result = analysis['nonexistent_name'].get_design_matrix()
