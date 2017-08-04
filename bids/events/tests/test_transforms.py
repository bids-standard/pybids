from bids.events.base import BIDSEventCollection
import pytest
from os.path import join, dirname
from bids import grabbids
import numpy as np
import pandas as pd


@pytest.fixture
def collection():
    path = join(dirname(grabbids.__file__), 'tests', 'data', 'ds005')
    evc = BIDSEventCollection(path)
    evc.read()
    return evc


def test_apply_rename(collection):
    dense_rt = collection.columns['RT'].to_dense()
    assert len(dense_rt.values) == 229280
    collection.apply('rename', 'RT', output='reaction_time')
    assert 'reaction_time' in collection.columns
    assert 'RT' not in collection.columns
    col = collection.columns['reaction_time']
    assert col.name == 'reaction_time'
    assert col.onsets.max() == 476


def test_apply_from_json(collection):
    ''' Same as test_apply_scale, but from json. '''
    path = join(dirname(__file__), 'data', 'transformations.json')
    collection.apply_from_json(path)
    groupby = collection['RT'].entities['event_file_id'].values
    z1 = collection['RT_Z'].values
    z2 = collection['RT'].values.groupby(
        groupby).apply(lambda x: (x - x.mean()) / x.std())
    assert np.allclose(z1, z2)


def test_apply_product(collection):
    c = collection
    collection.apply('product', cols=['parametric gain', 'gain'],
                     output='prod')
    res = c['prod'].values
    assert (res == c['parametric gain'].values * c['gain'].values).all()


def test_apply_scale(collection):
    collection.apply('scale', cols=['RT', 'parametric gain'],
                     output=['RT_Z', 'gain_Z'])
    groupby = collection['RT'].entities['event_file_id'].values
    z1 = collection['RT_Z'].values
    z2 = collection['RT'].values.groupby(
        groupby).apply(lambda x: (x - x.mean()) / x.std())
    assert np.allclose(z1, z2)


def test_apply_orthogonalize_dense(collection):
    pg_pre = collection['trial_type/parametric gain'].to_dense().values
    rt = collection['RT'].to_dense().values
    collection.apply('orthogonalize', cols='trial_type/parametric gain',
                     other='RT', dense=True)
    pg_post = collection['trial_type/parametric gain'].values
    vals = np.c_[rt.values, pg_pre.values, pg_post.values]
    df = pd.DataFrame(vals, columns=['rt', 'pre', 'post'])
    groupby = collection.dense_index['event_file_id']
    pre_r = df.groupby(groupby).apply(lambda x: x.corr().iloc[0, 1])
    post_r = df.groupby(groupby).apply(lambda x: x.corr().iloc[0, 2])
    assert (pre_r > 0.2).any()
    assert (post_r < 0.0001).all()


def test_apply_orthogonalize_sparse(collection):
    pg_pre = collection['parametric gain'].values
    rt = collection['RT'].values
    collection.apply('orthogonalize', cols='parametric gain',
                     other='RT')
    pg_post = collection['parametric gain'].values
    vals = np.c_[rt.values, pg_pre.values, pg_post.values]
    df = pd.DataFrame(vals, columns=['rt', 'pre', 'post'])
    groupby = collection['RT'].entities['event_file_id'].values
    pre_r = df.groupby(groupby).apply(lambda x: x.corr().iloc[0, 1])
    post_r = df.groupby(groupby).apply(lambda x: x.corr().iloc[0, 2])
    assert (pre_r > 0.2).any()
    assert (post_r < 0.0001).all()


def test_apply_split(collection):

    tmp = collection['RT'].clone(name='RT_2')
    collection['RT_2'] = tmp
    collection['RT_3'] = collection['RT'].clone(name='RT_3').to_dense()

    rt_pre_onsets = collection['RT'].onsets

    # Grouping SparseBIDSColumn by one column
    collection.apply('split', cols=['RT'], by=['respcat'])
    assert 'RT/0' in collection.columns.keys() and \
           'RT/-1' in collection.columns.keys()
    rt_post_onsets = np.r_[collection['RT/0'].onsets,
                           collection['RT/-1'].onsets,
                           collection['RT/1'].onsets]
    assert np.array_equal(rt_pre_onsets.sort(), rt_post_onsets.sort())

    # Grouping SparseBIDSColumn by multiple columns
    collection.apply('split', cols=['RT_2'], by=['respcat', 'loss'])
    tmp = collection['RT_2']
    assert 'RT_2/-1_13' in collection.columns.keys() and \
           'RT_2/1_13' in collection.columns.keys()

    # Grouping by DenseBIDSColumn
    collection.apply('split', cols='RT_3', by='respcat')
    assert 'RT_3/respcat[0.0]' in collection.columns.keys()
    assert len(collection['RT_3/respcat[0.0]'].values) == len(collection['RT_3'].values)

def test_resample_dense(collection):
    collection['RT'] = collection['RT'].to_dense()
    old_rt = collection['RT'].clone()
    collection.resample(50)
    assert len(old_rt.values) * 5 == len(collection['RT'].values)
    collection.resample(5, force_dense=True)
    assert len(old_rt.values) == len(collection['parametric gain'].values) * 2
