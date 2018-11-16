import pytest
from bids.layout import BIDSLayout
from bids.layout.layout import MetadataIndex
from os.path import join, abspath, sep
from bids.tests import get_test_data_path


# Fixture uses in the rest of the tests
@pytest.fixture(scope='module')
def layout():
    data_dir = join(get_test_data_path(), '7t_trt')
    return BIDSLayout(data_dir)


@pytest.fixture(scope='module')
def index(layout):
    return layout.metadata_index


def test_index_inits(index):
    assert hasattr(index, 'key_index')
    assert hasattr(index, 'file_index')
    assert not index.key_index
    assert not index.file_index

def test_get_metadata_caches_in_index(layout):
    targ = 'sub-04/ses-1/func/sub-04_ses-1_task-rest_acq-fullbrain_run-1_bold.nii.gz'
    targ = targ.split('/')
    targ = join(get_test_data_path(), '7t_trt', *targ)
    mdi = layout.metadata_index
    assert not mdi.file_index
    md = layout.get_metadata(targ)
    assert targ in mdi.file_index
    assert len(mdi.file_index) == 1
    assert 'CogAtlasID' in mdi.key_index
    assert 'RepetitionTime' in mdi.key_index

def test_searching_without_file_list_indexes_everything(index):
    res = index.search(nonexistent_key=2)
    assert not res
    keys = {'EchoTime2', 'EchoTime1', 'IntendedFor', 'CogAtlasID', 'EchoTime',
        'EffectiveEchoSpacing', 'PhaseEncodingDirection', 'RepetitionTime',
        'SliceEncodingDirection', 'SliceTiming', 'TaskName', 'StartTime',
        'SamplingFrequency', 'Columns', 'BIDSVersion', 'Name'}
    assert keys == set(index.key_index.keys())
    targ = 'sub-04/ses-1/func/sub-04_ses-1_task-rest_acq-fullbrain_run-1_bold.nii.gz'
    targ = targ.split('/')
    targ = join(get_test_data_path(), '7t_trt', *targ)
    assert targ in index.file_index
    assert index.file_index[targ]['EchoTime'] == 0.017

def test_search_with_no_args(index):
    with pytest.raises(ValueError) as exc:
        index.search()
    assert str(exc.value).startswith("At least one field")


def test_search_with_missing_keys(index):
    # Searching with invalid keys should return nothing
    res = index.search(keys_exist=['EchoTiming', 'Echolalia', 'EchoOneNiner'])
    assert res == []
    assert index.search(EchoTiming='eleventy') == []


def test_search_with_no_matching_value(index):
    results = index.search(EchoTime=0.017)
    assert results


def test_search_with_file_constraints(index, layout):
    files = layout.get(subject='03', return_type='file')
    results = index.search(EchoTime=0.017, files=files)
    assert len(results) == 4

def test_search_from_get(index, layout):
    results = layout.get(EchoTime=0.017, return_type='obj')
    assert len(results) == 40

    results = layout.get(EchoTime=0.017)
    assert len(results) == 40
