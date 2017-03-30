from bids.transform.base import BIDSEventCollection
from bids.transform.transform import BIDSTransformer
import pytest
from os.path import join, dirname
from bids import grabbids
import numpy as np
import pandas as pd


@pytest.fixture
def transformer():
    path = join(dirname(grabbids.__file__), 'tests', 'data', 'ds005')
    evc = BIDSEventCollection(path)
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


def test_apply_scale(transformer):
    t = transformer
    t.apply('scale', cols=['RT', 'parametric gain'], output=['RT_Z', 'gain_Z'])
    groupby = t.collection['RT'].entities['event_file_id'].values
    z1 = t.collection['RT_Z'].values
    z2 = t.collection['RT'].values.groupby(groupby).apply(lambda x: (x - x.mean())/ x.std())
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
