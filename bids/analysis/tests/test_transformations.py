from bids.analysis import BIDSVariableManager
from bids.analysis import transform
import pytest
from os.path import join, dirname, abspath
from bids import grabbids
import numpy as np
import pandas as pd


@pytest.fixture
def manager():
    mod_file = abspath(grabbids.__file__)
    path = join(dirname(mod_file), 'tests', 'data', 'ds005')
    manager = BIDSVariableManager(path)
    manager.load()
    return manager


def test_apply_rename(manager):
    dense_rt = manager.columns['RT'].to_dense()
    assert len(dense_rt.values) == 229280
    transform.rename(manager, 'RT', output='reaction_time')
    assert 'reaction_time' in manager.columns
    assert 'RT' not in manager.columns
    col = manager.columns['reaction_time']
    assert col.name == 'reaction_time'
    assert col.onsets.max() == 476


# def test_apply_from_json(manager):
#     ''' Same as test_apply_scale, but from json. '''
#     path = join(dirname(__file__), 'data', 'transformations.json')
#     manager.apply_from_json(path)
#     groupby = manager['RT'].entities['event_file_id'].values
#     z1 = manager['RT_Z'].values
#     z2 = manager['RT'].values.groupby(
#         groupby).apply(lambda x: (x - x.mean()) / x.std())
#     assert np.allclose(z1, z2)


def test_apply_product(manager):
    c = manager
    transform.product(manager, cols=['parametric gain', 'gain'], output='prod')
    res = c['prod'].values
    assert (res == c['parametric gain'].values * c['gain'].values).all()


def test_apply_scale(manager):
    transform.scale(manager, cols=['RT', 'parametric gain'],
                    output=['RT_Z', 'gain_Z'])
    groupby = manager['RT'].entities['event_file_id'].values
    z1 = manager['RT_Z'].values
    z2 = manager['RT'].values.groupby(
        groupby).apply(lambda x: (x - x.mean()) / x.std())
    assert np.allclose(z1, z2)


def test_apply_orthogonalize_dense(manager):
    transform.factor(manager, 'trial_type')
    pg_pre = manager['trial_type/parametric gain'].to_dense().values
    rt = manager['RT'].to_dense().values
    transform.orthogonalize(manager, cols='trial_type/parametric gain',
                            other='RT', dense=True)
    pg_post = manager['trial_type/parametric gain'].values
    vals = np.c_[rt.values, pg_pre.values, pg_post.values]
    df = pd.DataFrame(vals, columns=['rt', 'pre', 'post'])
    groupby = manager.dense_index['event_file_id']
    pre_r = df.groupby(groupby).apply(lambda x: x.corr().iloc[0, 1])
    post_r = df.groupby(groupby).apply(lambda x: x.corr().iloc[0, 2])
    assert (pre_r > 0.2).any()
    assert (post_r < 0.0001).all()


def test_apply_orthogonalize_sparse(manager):
    pg_pre = manager['parametric gain'].values
    rt = manager['RT'].values
    transform.orthogonalize(manager, cols='parametric gain', other='RT')
    pg_post = manager['parametric gain'].values
    vals = np.c_[rt.values, pg_pre.values, pg_post.values]
    df = pd.DataFrame(vals, columns=['rt', 'pre', 'post'])
    groupby = manager['RT'].entities['event_file_id'].values
    pre_r = df.groupby(groupby).apply(lambda x: x.corr().iloc[0, 1])
    post_r = df.groupby(groupby).apply(lambda x: x.corr().iloc[0, 2])
    assert (pre_r > 0.2).any()
    assert (post_r < 0.0001).all()


def test_apply_split(manager):

    tmp = manager['RT'].clone(name='RT_2')
    manager['RT_2'] = tmp
    manager['RT_3'] = manager['RT'].clone(name='RT_3').to_dense()

    rt_pre_onsets = manager['RT'].onsets

    # Grouping SparseBIDSColumn by one column
    transform.split(manager, ['RT'], ['respcat'])
    assert 'RT/0' in manager.columns.keys() and \
           'RT/-1' in manager.columns.keys()
    rt_post_onsets = np.r_[manager['RT/0'].onsets,
                           manager['RT/-1'].onsets,
                           manager['RT/1'].onsets]
    assert np.array_equal(rt_pre_onsets.sort(), rt_post_onsets.sort())

    # Grouping SparseBIDSColumn by multiple columns
    transform.split(manager, cols=['RT_2'], by=['respcat', 'loss'])
    tmp = manager['RT_2']
    assert 'RT_2/-1_13' in manager.columns.keys() and \
           'RT_2/1_13' in manager.columns.keys()

    # Grouping by DenseBIDSColumn
    transform.split(manager, cols='RT_3', by='respcat')
    assert 'RT_3/respcat[0.0]' in manager.columns.keys()
    assert len(manager['RT_3/respcat[0.0]'].values) == len(manager['RT_3'].values)


def test_resample_dense(manager):
    manager['RT'] = manager['RT'].to_dense()
    old_rt = manager['RT'].clone()
    manager.resample(50, in_place=True)
    assert len(old_rt.values) * 5 == len(manager['RT'].values)
    # Should work after explicitly converting categoricals
    transform.factor(manager, 'trial_type')
    manager.resample(5, force_dense=True, in_place=True)
    assert len(old_rt.values) == len(manager['parametric gain'].values) * 2


def test_threshold(manager):
    old_pg = manager['parametric gain']
    orig_vals = old_pg.values

    manager['pg'] = old_pg.clone()
    transform.threshold(manager, 'pg', threshold=0.2, binarize=True)
    assert manager.columns['pg'].values.sum() == (orig_vals >= 0.2).sum()

    manager['pg'] = old_pg.clone()
    transform.threshold(manager, 'pg', threshold=0.2, binarize=False)
    assert manager.columns['pg'].values.sum() != (orig_vals >= 0.2).sum()
    assert (manager.columns['pg'].values >= 0.2).sum() == (orig_vals >= 0.2).sum()

    manager['pg'] = old_pg.clone()
    transform.threshold(manager, 'pg', threshold=-0.1, binarize=True,
                        signed=False, above=False)
    n = np.logical_and(orig_vals <= 0.1, orig_vals >= -0.1).sum()
    assert manager.columns['pg'].values.sum() == n


def test_assign(manager):
    transform.assign(manager, 'parametric gain', target='RT',
                     target_attr='onset', output='test1')
    t1 = manager['test1']
    pg = manager['parametric gain']
    rt = manager['RT']
    assert np.array_equal(t1.onsets, pg.values.values)
    assert np.array_equal(t1.durations, rt.durations)
    assert np.array_equal(t1.values.values, rt.values.values)

    transform.assign(manager, 'RT', target='parametric gain',
                     input_attr='onset', target_attr='amplitude',
                     output='test2')
    t2 = manager['test2']
    assert np.array_equal(t2.values.values, rt.onsets)
    assert np.array_equal(t2.onsets, pg.onsets)
    assert np.array_equal(t2.durations, pg.durations)


def test_copy(manager):
    transform.copy(manager, 'RT', output='RT_copy')
    assert 'RT_copy' in manager.columns.keys()
    assert np.array_equal(manager['RT'].values.values, manager['RT_copy'].values.values)


def test_regex_column_expansion(manager):
    # Should fail because two output values are required following expansion
    with pytest.raises(Exception):
        transform.copy(manager, 'resp', regex_columns='cols')

    transform.copy(manager, 'resp', output_suffix='_copy',
                   regex_columns='cols')
    assert 'respnum_copy' in manager.columns.keys()
    assert 'respcat_copy' in manager.columns.keys()
    assert np.array_equal(manager['respcat'].values.values,
                          manager['respcat_copy'].values.values)
    assert np.array_equal(manager['respnum'].values.values,
                          manager['respnum_copy'].values.values)


def test_factor(manager):
    transform.factor(manager, 'trial_type')
    assert 'trial_type/parametric gain' in manager.columns.keys()
    pg = manager.columns['trial_type/parametric gain']
    assert pg.values.unique() == [1]


def test_filter(manager):
    orig = manager['parametric gain'].clone()
    q = "parametric gain > 0"
    transform.filter(manager, 'parametric gain', query=q)
    assert len(orig.values) == 2 * len(manager['parametric gain'].values)
    assert np.all(manager['parametric gain'].values > 0)

    orig = manager['RT'].clone()
    transform.filter(manager, 'RT', by='parametric gain', query=q)
    assert len(orig.values) == 2 * len(manager['RT'].values)
