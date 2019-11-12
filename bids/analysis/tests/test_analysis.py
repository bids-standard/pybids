from os.path import join
from bids.analysis import Analysis
from bids.analysis.analysis import ContrastInfo, DesignMatrixInfo
from bids.layout import BIDSLayout
from bids.tests import get_test_data_path
import numpy as np
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
        assert isinstance(dmi, DesignMatrixInfo)
        assert dmi._fields == ('sparse', 'dense', 'entities')
        assert hasattr(dmi.sparse, 'shape')
        assert dmi.dense is None
        assert isinstance(dmi.entities, dict)


def test_get_design_matrix_arguments(analysis):
    kwargs = dict(run=1, subject='01', sparse=True)
    result = analysis['run'].get_design_matrix(**kwargs)
    result = result[0]
    assert result.sparse.shape == (172, 9)
    assert result.dense is None

    kwargs = dict(run=1, subject='01', mode='dense', force=False)
    result = analysis['run'].get_design_matrix(**kwargs)[0]
    assert result.sparse is None
    assert result.dense is None

    kwargs = dict(run=1, subject='01', mode='dense', force=True,
                  sampling_rate='highest')
    result = analysis['run'].get_design_matrix(**kwargs)[0]
    assert result.sparse is None
    assert result.dense.shape == (4800, 10)

    kwargs = dict(run=1, subject='01', mode='dense', force=True,
                  sampling_rate='TR')
    result = analysis['run'].get_design_matrix(**kwargs)[0]
    assert result.sparse is None
    assert result.dense.shape == (240, 10)

    kwargs = dict(run=1, subject='01', mode='dense', force=True,
                  sampling_rate=0.5)
    result = analysis['run'].get_design_matrix(**kwargs)[0]
    assert result.sparse is None
    assert result.dense.shape == (240, 10)

    # format='long' should be ignored for dense output
    kwargs = dict(run=1, subject='01', mode='dense', force=True,
                  format='long', entities=False)
    result = analysis['run'].get_design_matrix(**kwargs)[0]
    assert result.sparse is None
    assert result.dense.shape == (240, 1)

    kwargs = dict(run=1, subject='01', mode='sparse', format='wide',
                  entities=False)
    result = analysis['run'].get_design_matrix(**kwargs)[0]
    assert result.dense is None
    assert result.sparse.shape == (86, 4)


def test_first_level_sparse_design_matrix(analysis):
    result = analysis['run'].get_design_matrix(subject=['01'])
    assert len(result) == 3
    df = result[0].sparse
    assert df.shape == (172, 9)
    assert df['condition'].nunique() == 2
    assert set(result[0][0].columns) == {'amplitude', 'onset', 'duration',
                                         'condition', 'subject', 'run',
                                         'task', 'datatype', 'suffix'}


def test_post_first_level_sparse_design_matrix(analysis):

    result = analysis['session'].get_design_matrix(entities=False)
    assert len(result) == 2
    assert len(result[0]) == 3
    assert result[0].sparse.shape == (9, 2)
    assert result[0].entities == {
        'subject': '01',
        'task': 'mixedgamblestask',
        'datatype': 'func',
        'suffix': 'bold'}

    # Participant level and also check integer-based indexing
    result = analysis['participant'].get_design_matrix()
    assert len(result) == 2
    assert analysis[2].name == 'participant'

    # Dataset level
    result = analysis['group'].get_design_matrix()
    assert len(result) == 1
    data = result[0].sparse
    assert len(data) == 10
    assert data['subject'].nunique() == 2

    # # Make sure columns from different levels exist
    varset = {'sex', 'age', 'RT-trial_type'}
    assert not (varset - set(data['condition'].unique()))

    # Calling an invalid level name should raise an exception
    with pytest.raises(KeyError):
        result = analysis['nonexistent_name'].get_design_matrix()


def test_contrast_info(analysis):
    contrast_lists = analysis['run'].get_contrasts(subject='01')
    assert len(contrast_lists) == 3
    for cl in contrast_lists:
        assert len(cl) == 3
        cl = [c for c in cl if c.type == 't']
        assert set([c.name for c in cl]) == {'RT', 'RT-trial_type'}
        assert set([c.type for c in cl]) == {'t'}
        assert cl[0].weights.columns.tolist() == ['RT', 'trial_type']
        assert cl[1].weights.columns.tolist() == ['RT']
        assert np.array_equal(cl[0].weights.values, np.array([[1, -1]]))
        assert np.array_equal(cl[1].weights.values, np.array([[1]]))
        assert isinstance(cl[0], ContrastInfo)
        assert cl[0]._fields == ('name', 'weights', 'type', 'entities')


def test_contrast_info_with_specified_variables(analysis):
    varlist = ['RT', 'dummy']
    contrast_lists = analysis['run'].get_contrasts(subject='01',
                                                   variables=varlist)
    assert len(contrast_lists) == 3
    for cl in contrast_lists:
        assert len(cl) == 3
        cl = [c for c in cl if c.type == 't']
        assert set([c.name for c in cl]) == {'RT', 'RT-trial_type'}
        assert set([c.type for c in cl]) == {'t'}
        for c in cl:
            assert c.weights.columns.tolist() == ['RT', 'dummy']
            assert np.array_equal(c.weights.values, np.array([[1, 0]]))
        assert isinstance(cl[0], ContrastInfo)
        assert cl[0]._fields == ('name', 'weights', 'type', 'entities')


def test_contrast_info_F_contrast(analysis):
    contrast_lists = analysis['run'].get_contrasts(subject='01',
                                                   names=["crummy-F"])
    assert len(contrast_lists) == 3
    for cl in contrast_lists:
        assert len(cl) == 1
        c = cl[0]
        assert c.name == "crummy-F"
        assert c.type == 'F'
        assert c.weights.columns.tolist() == ['RT', 'trial_type']
        assert np.array_equal(c.weights.values, np.array([[1, 0], [0, 1]]))
        assert isinstance(c, ContrastInfo)
        assert c._fields == ('name', 'weights', 'type', 'entities')


def test_dummy_contrasts(analysis):
    names = [c.name for c in analysis['run'].get_contrasts(subject='01')[0]]
    session = analysis['session'].get_contrasts(subject='01')[0]
    for cl in session:
        assert cl.type == 'FEMA'
        assert cl.name in names

    participant = analysis['participant'].get_contrasts(subject='01')[0]
    assert len(participant) == 3
    for cl in participant:
        assert cl.type == 'FEMA'
        assert cl.name in names

    group = analysis['group'].get_contrasts()[0]
    group_names = []
    for cl in group:
        assert cl.type == 't'
        group_names.append(cl.name)

    assert set(names) < set(group_names)
