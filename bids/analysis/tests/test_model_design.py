from os.path import join
import json

import pandas as pd
import numpy as np
import pytest

from bids.variables import BIDSVariableCollection
from bids.analysis.model_design import GLMMSpec, TransformerManager
from bids.layout import BIDSLayout
from bids.tests import get_test_data_path


def test_bvc_to_model_design():
    layout_path = join(get_test_data_path(), 'ds005')
    json_file = join(layout_path, 'models', 'ds-005_type-mfx_model.json')
    spec = json.load(open(json_file, 'r'))

    # Make a fake collection at dataset level
    df = pd.DataFrame({
        'subject': np.repeat(np.arange(10), 4),
        'run': np.tile(np.arange(4), 10),
        'age': np.repeat(np.random.normal(40, 10, 10).astype(int), 4)
    })
    entities = df[['subject', 'run']]
    collection = BIDSVariableCollection.from_df(df, entities=entities)

    tm = TransformerManager()
    collection = tm.transform(collection, spec['Steps'][1]['Transformations'])
    md = GLMMSpec.from_collection(collection, spec['Steps'][1]['Model'])

    assert len(md.terms) == 2
    assert md.terms['age'].values.shape == (40, )
    assert md.terms['SubjectSlopes'].values.shape == (40, 10)
    index_vec = md.terms['SubjectSlopes'].index_vec
    assert index_vec.shape == (40,)
    assert np.array_equal(np.sort(np.unique(index_vec)), np.arange(10) + 1)
