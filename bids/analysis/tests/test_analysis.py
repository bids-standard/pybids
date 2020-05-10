from os.path import join

import numpy as np
import pytest

from bids.analysis import Analysis
from bids.analysis.analysis import ContrastInfo
from bids.layout import BIDSLayout
from bids.tests import get_test_data_path
from bids.variables import BIDSVariableCollection


@pytest.fixture
def analysis():
    layout_path = join(get_test_data_path(), 'ds005')
    layout = BIDSLayout(layout_path)
    json_file = join(layout_path, 'models', 'ds-005_type-test_model.json')
    analysis = Analysis(layout, json_file)
    analysis.setup(scan_length=480, subject=['01', '02'])
    return analysis


def test_first_level_sparse_design_matrix(analysis):
    collections = analysis['run'].get_collections(subject=['01'])
    assert len(collections) == 3
    df = collections[0].to_df(format='long')
    assert df.shape == (172, 9)
    assert df['condition'].nunique() == 2
    assert set(df.columns) == {'amplitude', 'onset', 'duration',
                               'condition', 'subject', 'run',
                               'task', 'datatype', 'suffix'}


def test_post_first_level_sparse_design_matrix(analysis):

    collections = analysis['session'].get_collections()
    assert len(collections) == 2
    result = collections[0].to_df(format='long')
    assert result.shape == (9, 11)
    result = collections[0].to_df(format='long', entities=False)
    assert result.shape == (9, 2)
    entities = {
        # 'subject': '01',  # PY35
        'task': 'mixedgamblestask',
        'datatype': 'func',
        'suffix': 'bold'}
    assert not set(entities.keys()) - set(collections[0].entities.keys())
    assert not set(entities.values()) - set(collections[0].entities.values())
    # PY35
    assert 'subject' in collections[0].entities
    assert collections[0].entities['subject'] in ('01', '02')

    # Participant level and also check integer-based indexing
    collections = analysis['participant'].get_collections()
    assert len(collections) == 2
    assert analysis[2].name == 'participant'

    # Dataset level
    collections = analysis['group'].get_collections()
    assert len(collections) == 1
    data = collections[0].to_df(format='long')
    assert len(data) == 10
    assert data['subject'].nunique() == 2

    # # Make sure columns from different levels exist
    varset = {'sex', 'age', 'RT-trial_type'}
    assert not (varset - set(data['condition'].unique()))

    # Calling an invalid level name should raise an exception
    with pytest.raises(KeyError):
        result = analysis['nonexistent_name'].to_df()


def test_step_get_collections(analysis):
    collections = analysis['run'].get_collections(subject='01')
    assert len(collections) == 3
    assert isinstance(collections[0], BIDSVariableCollection)


def test_contrast_info(analysis):
    colls = analysis['run'].get_collections(subject='01')
    contrast_lists = [analysis['run'].get_contrasts(c) for c in colls]
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
    colls = analysis['run'].get_collections(subject='01')
    contrast_lists = [analysis['run'].get_contrasts(c, variables=varlist)
                      for c in colls]
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
    colls = analysis['run'].get_collections(subject='01')
    contrast_lists = [analysis['run'].get_contrasts(c, names=["crummy-F"])
                      for c in colls]
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
    collection = analysis['run'].get_collections(subject='01')[0]
    names = [c.name for c in analysis['run'].get_contrasts(collection)]

    collections = analysis['session'].get_collections()
    collection = analysis['session'].get_collections(subject='01')[0]
    session = analysis['session'].get_contrasts(collection)
    for cl in session:
        assert cl.type == 'FEMA'
        assert cl.name in names

    collection = analysis['participant'].get_collections(subject='01')[0]
    participant = analysis['participant'].get_contrasts(collection)
    assert len(participant) == 3
    for cl in participant:
        assert cl.type == 'FEMA'
        assert cl.name in names

    collection = analysis['group'].get_collections()[0]
    group = analysis['group'].get_contrasts(collection)
    group_names = []
    for cl in group:
        assert cl.type == 't'
        group_names.append(cl.name)

    assert set(names) < set(group_names)


def test_get_model_spec(analysis):
    collection = analysis['run'].get_collections(subject='01', run=1)[0]
    model_spec = analysis['run'].get_model_spec(collection, 'TR')
    assert model_spec.__class__.__name__ == 'GLMMSpec'
    assert model_spec.X.shape == (240, 1)
    assert model_spec.Z is None
    assert {'RT'} == set(model_spec.terms.keys())
