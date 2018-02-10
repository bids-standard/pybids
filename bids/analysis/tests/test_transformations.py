# from bids.analysis.variables import load_variables
from bids.analysis import transformations as transform
from bids.grabbids import BIDSLayout
import pytest
from os.path import join
from bids.tests import get_test_data_path
import numpy as np
import pandas as pd


@pytest.fixture
def collection():
    layout_path = join(get_test_data_path(), 'ds005')
    layout = BIDSLayout(layout_path)
    collection = layout.get_collections('run', types=['events'],
                                        scan_length=480, merge=True,
                                        sampling_rate=10)
    return collection


def test_rename(collection):
    dense_rt = collection.variables['RT'].to_dense(10)
    assert len(dense_rt.values) == 230400
    transform.rename(collection, 'RT', output='reaction_time')
    assert 'reaction_time' in collection.variables
    assert 'RT' not in collection.variables
    col = collection.variables['reaction_time']
    assert col.name == 'reaction_time'
    assert col.onset.max() == 476


def test_product(collection):
    c = collection
    transform.product(collection, variables=['parametric gain', 'gain'],
                      output='prod')
    res = c['prod'].values
    assert (res == c['parametric gain'].values * c['gain'].values).all()


def test_scale(collection):
    transform.scale(collection, variables=['RT', 'parametric gain'],
                    output=['RT_Z', 'gain_Z'])
    ents = collection['RT'].index
    groupby = pd.core.groupby._get_grouper(ents, ['run', 'subject'])[0]
    z1 = collection['RT_Z'].values
    z2 = collection['RT'].values.groupby(
        groupby).apply(lambda x: (x - x.mean()) / x.std())
    assert np.allclose(z1, z2)


def test_orthogonalize_dense(collection):
    transform.factor(collection, 'trial_type', sep='/')

    # Store pre-orth variables needed for tests
    pg_pre = collection['trial_type/parametric gain'].to_dense(10)
    rt = collection['RT'].to_dense(10)

    # Orthogonalize and store result
    transform.orthogonalize(collection, variables='trial_type/parametric gain',
                            other='RT', dense=True, groupby=['run', 'subject'])
    pg_post = collection['trial_type/parametric gain']

    # Verify that the to_dense() calls result in identical indexing
    ent_cols = ['subject', 'run']
    assert pg_pre.to_df()[ent_cols].equals(rt.to_df()[ent_cols])
    assert pg_post.to_df()[ent_cols].equals(rt.to_df()[ent_cols])

    vals = np.c_[rt.values, pg_pre.values, pg_post.values]
    df = pd.DataFrame(vals, columns=['rt', 'pre', 'post'])
    ents = rt.index
    groupby = pd.core.groupby._get_grouper(ents, ['run', 'subject'])[0]
    pre_r = df.groupby(groupby).apply(lambda x: x.corr().iloc[0, 1])
    post_r = df.groupby(groupby).apply(lambda x: x.corr().iloc[0, 2])
    assert (pre_r > 0.2).any()
    assert (post_r < 0.0001).all()


def test_orthogonalize_sparse(collection):
    pg_pre = collection['parametric gain'].values
    rt = collection['RT'].values
    transform.orthogonalize(collection, variables='parametric gain',
                            other='RT')
    pg_post = collection['parametric gain'].values
    vals = np.c_[rt.values, pg_pre.values, pg_post.values]
    df = pd.DataFrame(vals, columns=['rt', 'pre', 'post'])
    ents = collection['RT'].index
    groupby = pd.core.groupby._get_grouper(ents, ['run', 'subject'])[0]
    pre_r = df.groupby(groupby).apply(lambda x: x.corr().iloc[0, 1])
    post_r = df.groupby(groupby).apply(lambda x: x.corr().iloc[0, 2])
    assert (pre_r > 0.2).any()
    assert (post_r < 0.0001).all()


def test_split(collection):

    orig = collection['RT'].clone(name='RT_2')
    collection['RT_2'] = orig.clone()
    collection['RT_3'] = collection['RT'].clone(name='RT_3').to_dense(10)

    rt_pre_onsets = collection['RT'].onset

    # Grouping SparseEventVariable by one column
    transform.split(collection, ['RT'], ['respcat'])
    assert 'RT.0' in collection.variables.keys() and \
           'RT.-1' in collection.variables.keys()
    rt_post_onsets = np.r_[collection['RT.0'].onset,
                           collection['RT.-1'].onset,
                           collection['RT.1'].onset]
    assert np.array_equal(rt_pre_onsets.sort(), rt_post_onsets.sort())

    # Grouping SparseEventVariable by multiple columns
    transform.split(collection, variables=['RT_2'], by=['respcat', 'loss'])
    assert 'RT_2.-1_13' in collection.variables.keys() and \
           'RT_2.1_13' in collection.variables.keys()

    # Grouping by DenseEventVariable
    transform.split(collection, variables='RT_3', by='respcat',
                    drop_orig=False)
    targets = ['RT_3.respcat[-1]', 'RT_3.respcat[0]', 'RT_3.respcat[1]']
    assert not set(targets) - set(collection.variables.keys())
    assert collection['respcat'].values.nunique() == 3
    n_dense = len(collection['RT_3'].values)
    assert len(collection['RT_3.respcat[-1]'].values) == n_dense

    # Grouping by entities in the index
    collection['RT_4'] = orig.clone(name='RT_4')
    transform.split(collection, variables=['RT_4'], by=['respcat', 'run'])
    assert 'RT_4.-1_3' in collection.variables.keys()


def test_resample_dense(collection):
    collection['RT'] = collection['RT'].to_dense(10)
    old_rt = collection['RT'].clone()
    collection.resample(50, in_place=True)
    assert len(old_rt.values) * 5 == len(collection['RT'].values)
    # Should work after explicitly converting categoricals
    transform.factor(collection, 'trial_type')
    collection.resample(5, force_dense=True, in_place=True)
    assert len(old_rt.values) == len(collection['parametric gain'].values) * 2


def test_threshold(collection):
    old_pg = collection['parametric gain']
    orig_vals = old_pg.values

    collection['pg'] = old_pg.clone()
    transform.threshold(collection, 'pg', threshold=0.2, binarize=True)
    assert collection.variables['pg'].values.sum() == (orig_vals >= 0.2).sum()

    collection['pg'] = old_pg.clone()
    transform.threshold(collection, 'pg', threshold=0.2, binarize=False)
    assert collection.variables['pg'].values.sum() != (orig_vals >= 0.2).sum()
    coll_sum = (collection.variables['pg'].values >= 0.2).sum()
    assert coll_sum == (orig_vals >= 0.2).sum()

    collection['pg'] = old_pg.clone()
    transform.threshold(collection, 'pg', threshold=-0.1, binarize=True,
                        signed=False, above=False)
    n = np.logical_and(orig_vals <= 0.1, orig_vals >= -0.1).sum()
    assert collection.variables['pg'].values.sum() == n


def test_assign(collection):
    transform.assign(collection, 'parametric gain', target='RT',
                     target_attr='onset', output='test1')
    t1 = collection['test1']
    pg = collection['parametric gain']
    rt = collection['RT']
    assert np.array_equal(t1.onset, pg.values.values)
    assert np.array_equal(t1.duration, rt.duration)
    assert np.array_equal(t1.values.values, rt.values.values)

    transform.assign(collection, 'RT', target='parametric gain',
                     input_attr='onset', target_attr='amplitude',
                     output='test2')
    t2 = collection['test2']
    assert np.array_equal(t2.values.values, rt.onset)
    assert np.array_equal(t2.onset, pg.onset)
    assert np.array_equal(t2.duration, pg.duration)


def test_copy(collection):
    transform.copy(collection, 'RT', output='RT_copy')
    assert 'RT_copy' in collection.variables.keys()
    assert np.array_equal(collection['RT'].values.values,
                          collection['RT_copy'].values.values)


def test_regex_variable_expansion(collection):
    # Should fail because two output values are required following expansion
    with pytest.raises(Exception):
        transform.copy(collection, 'resp', regex_variables='variables')

    transform.copy(collection, 'resp', output_suffix='_copy',
                   regex_variables='variables')
    assert 'respnum_copy' in collection.variables.keys()
    assert 'respcat_copy' in collection.variables.keys()
    assert np.array_equal(collection['respcat'].values.values,
                          collection['respcat_copy'].values.values)
    assert np.array_equal(collection['respnum'].values.values,
                          collection['respnum_copy'].values.values)


def test_factor(collection):
    transform.factor(collection, 'trial_type', sep='@')
    assert 'trial_type@parametric gain' in collection.variables.keys()
    pg = collection.variables['trial_type@parametric gain']
    assert pg.values.unique() == [1]


def test_filter(collection):
    orig = collection['parametric gain'].clone()
    q = "parametric gain > 0"
    transform.filter(collection, 'parametric gain', query=q)
    assert len(orig.values) == 2 * len(collection['parametric gain'].values)
    assert np.all(collection['parametric gain'].values > 0)

    orig = collection['RT'].clone()
    transform.filter(collection, 'RT', by='parametric gain', query=q)
    assert len(orig.values) == 2 * len(collection['RT'].values)


def test_select(collection):
    keep = ['RT', 'parametric gain', 'respcat']
    transform.select(collection, keep)
    assert set(collection.variables.keys()) == set(keep)
