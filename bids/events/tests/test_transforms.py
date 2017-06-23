from bids.events.base import BIDSEventCollection, BIDSTransformer
import pytest
from os.path import join, dirname
from bids import grabbids
import numpy as np
import pandas as pd


@pytest.fixture
def transformer():
    path = join(dirname(grabbids.__file__), 'tests', 'data', 'ds005')
    evc = BIDSEventCollection(path)
    evc.read()
    return BIDSTransformer(evc)


def test_apply_rename(transformer):
    t = transformer
    coll = t.collection
    dense_rt = coll.columns['RT'].to_dense(t)
    assert len(dense_rt.values) == 229280
    t.apply('rename', 'RT', output='reaction_time')
    assert 'reaction_time' in coll.columns
    assert 'RT' not in coll.columns
    col = coll.columns['reaction_time']
    assert col.name == 'reaction_time'
    assert col.onsets.max() == 476


def test_apply_from_json(transformer):
    pass


def test_apply_product(transformer):
    t = transformer
    c = t.collection
    t.apply('product', cols=['parametric gain', 'gain'], output='prod')
    res = c['prod'].values
    assert (res == c['parametric gain'].values * c['gain'].values).all()


def test_apply_scale(transformer):
    t = transformer
    t.apply('scale', cols=['RT', 'parametric gain'], output=['RT_Z', 'gain_Z'])
    groupby = t.collection['RT'].entities['event_file_id'].values
    z1 = t.collection['RT_Z'].values
    z2 = t.collection['RT'].values.groupby(
        groupby).apply(lambda x: (x - x.mean()) / x.std())
    assert np.allclose(z1, z2)


def test_apply_orthogonalize(transformer):
    t = transformer
    coll = t.collection
    pg_pre = coll['trial_type/parametric gain'].to_dense(t).values
    rt = coll['RT'].to_dense(t).values
    t.apply('orthogonalize', cols='trial_type/parametric gain', other='RT',
            dense=True)
    pg_post = coll['trial_type/parametric gain'].values
    vals = np.c_[rt.values, pg_pre.values, pg_post.values]
    df = pd.DataFrame(vals, columns=['rt', 'pre', 'post'])
    groupby = t.dense_index['event_file_id']
    pre_r = df.groupby(groupby).apply(lambda x: x.corr().iloc[0, 1])
    post_r = df.groupby(groupby).apply(lambda x: x.corr().iloc[0, 2])
    assert (pre_r > 0.2).any()
    assert (post_r < 0.0001).all()


def test_apply_split(transformer):
    t = transformer
    coll = t.collection
    rt_pre_onsets = coll['RT'].onsets

    # Grouping SparseBIDSColumn by one column
    t.apply('split', cols=['RT'], by=['respcat'])
    assert 'RT/0' in coll.columns.keys() and 'RT/-1' in coll.columns.keys()
    rt_post_onsets = np.r_[coll['RT/0'].onsets, coll['RT/-1'].onsets,
                           coll['RT/1'].onsets]
    assert np.array_equal(rt_pre_onsets.sort(), rt_post_onsets.sort())

    # Grouping SparseBIDSColumn by multiple columns
    t.apply('split', cols=['RT'], by=['respcat', 'loss'])
    assert 'RT/-1_13' in coll.columns.keys() and 'RT/1_13' in coll.columns.keys()

    # Grouping DenseBIDSColumn
    coll['RT_2'] = coll['RT'].clone(name='RT_2').to_dense(t)
    t.apply('split', cols='RT_2', by='respcat')
    assert 'RT_2/respcat[0.0]' in coll.columns.keys()
    assert len(coll['RT_2/respcat[0.0]'].values) == len(coll['RT_2'].values)
