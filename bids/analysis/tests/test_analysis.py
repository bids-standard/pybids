from os.path import join
from bids.analysis import Analysis
from bids.analysis.analysis import ContrastInfo
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


def test_get_design_matrix_arguments(analysis):
    kwargs = dict(run=1, subject='01', sparse=True)
    nodes = analysis['run'].get_nodes(**kwargs)
    assert len(nodes) == 1
    result = nodes[0].get_design_matrix(mode='sparse')
    assert result.shape == (172, 9)

    result = nodes[0].get_design_matrix(mode='dense', force=False)
    assert result is None

    result = nodes[0].get_design_matrix(mode='dense', force=True,
                                        sampling_rate='highest')
    assert result.shape == (4800, 10)

    result = nodes[0].get_design_matrix(mode='dense', force=True,
                                        sampling_rate='TR')
    assert result.shape == (240, 10)

    result = nodes[0].get_design_matrix(mode='dense', force=True,
                                        sampling_rate=0.5)
    assert result.shape == (240, 10)

    # format='long' should be ignored for dense output
    result = nodes[0].get_design_matrix(mode='dense', force=True, format='long',
                                        entities=False)
    assert result.shape == (240, 1)

    result = nodes[0].get_design_matrix(mode='sparse', format='wide',
                                        entities=False)
    assert result.shape == (86, 4)


def test_first_level_sparse_design_matrix(analysis):
    nodes = analysis['run'].get_nodes(subject=['01'])
    assert len(nodes) == 3
    df = nodes[0].get_design_matrix(mode='sparse')
    assert df.shape == (172, 9)
    assert df['condition'].nunique() == 2
    assert set(df.columns) == {'amplitude', 'onset', 'duration',
                                         'condition', 'subject', 'run',
                                         'task', 'datatype', 'suffix'}


def test_post_first_level_sparse_design_matrix(analysis):

    nodes = analysis['session'].get_nodes()
    assert len(nodes) == 2
    result = nodes[0].get_design_matrix(mode='sparse')
    assert result.shape == (9, 7)
    result = nodes[0].get_design_matrix(mode='sparse', entities=False)
    assert result.shape == (9, 2)
    assert nodes[0].entities == {
        'subject': '01',
        'task': 'mixedgamblestask',
        'datatype': 'func',
        'suffix': 'bold'}

    # Participant level and also check integer-based indexing
    nodes = analysis['participant'].get_nodes()
    assert len(nodes) == 2
    assert analysis[2].name == 'participant'

    # Dataset level
    nodes = analysis['group'].get_nodes()
    assert len(nodes) == 1
    data = nodes[0].get_design_matrix(mode='sparse')
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
