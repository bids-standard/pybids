from bids.transform.base import BIDSEventCollection
from bids.transform.transform import BIDSTransformer
import pytest
from os.path import join, dirname
from bids import grabbids
import numpy as np


@pytest.fixture
def transformer():
    path = join(dirname(grabbids.__file__), 'tests', 'data', 'ds005')
    evc = BIDSEventCollection(path)
    return BIDSTransformer(evc)


def test_apply_rename(transformer):
    t = transformer
    dense_rt = t.columns['RT'].to_dense(t)
    assert len(dense_rt.values) == 229280
    t.apply('rename', 'RT', 'reaction_time')
    assert 'reaction_time' in t.columns
    assert 'RT' not in t.columns
    col = t.columns['reaction_time']
    assert col.name == 'reaction_time'
    assert col.onsets.max() == 476


def test_apply_from_json(transformer):
    pass


def test_apply_standardize(transformer):
    t = transformer
    t.apply('standardize', cols=['RT', 'parametric gain'], output=['RT_Z', 'gain_Z'])
    orig = t.collection['RT'].values
    z = t.collection['RT_Z'].values
    assert np.allclose(z, (orig - orig.mean()) / orig.std())

def test_apply_orthogonalize(transformer):
    t = transformer
    t.apply('orthogonalize', cols='parametric gain', other='RT')