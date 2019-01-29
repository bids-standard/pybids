import pytest
from bids.layout import BIDSLayout
from os.path import join, abspath, sep
from bids.tests import get_test_data_path


@pytest.fixture(scope='module')
def layout():
    data_dir = join(get_test_data_path(), '7t_trt')
    return BIDSLayout(data_dir)

@pytest.mark.skip(reason="Disabled until path-building is added again following refactoring.")
def test_bold_construction(layout):
    ents = dict(subject='01', run=1, task='rest', suffix='bold')
    assert layout.build_path(ents) == "sub-01/func/sub-01_task-rest_run-1_bold.nii.gz"
    ents['acquisition'] = 'random'
    assert layout.build_path(ents) == "sub-01/func/sub-01_task-rest_acq-random_run-1_bold.nii.gz"