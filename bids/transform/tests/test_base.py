from bids.transform.base import (SparseBIDSColumn, BIDSEventCollection,
                                 BIDSEventFile, BIDSTransformer)
import pytest
from os.path import join, dirname


@pytest.fixture
def bids_event_collection():
    from bids import grabbids
    path = join(dirname(grabbids.__file__), 'tests', 'data', 'ds005')
    return BIDSEventCollection(path)


def test_collection(bids_event_collection):
    ''' Integration test for BIDSEventCollection initialization. '''

    bec = bids_event_collection

    # Test collection attributes
    assert bec.condition_column == 'trial_type'

    # Test that event files are loaded properly
    assert len(bec.event_files) == 48
    ef = bec.event_files[0]
    assert isinstance(ef, BIDSEventFile)
    assert ef.entities['task'] == 'mixedgamblestask'
    assert ef.entities['subject'] == '01'

    # Test extracted columns
    col_keys = bec.columns.keys()
    assert set(col_keys) == {'RT', 'gain', 'respnum', 'PTval', 'loss',
                             'respcat', 'parametric gain'}
    col = bec.columns['RT']
    assert isinstance(col, SparseBIDSColumn)
    assert col.collection == bec
    assert col.name == 'RT'
    assert col.onsets.max() == 476
    assert (col.durations == 3).all()
    assert len(col.durations) == 4096
    ents = col.entities
    assert (ents['task'] == 'mixedgamblestask').all()
    assert set(ents.columns) == {'task', 'subject', 'run', 'event_file_id'}
