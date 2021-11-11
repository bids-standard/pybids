import pytest

from os.path import join

from bids.layout import BIDSLayout
from bids.reports import parameters
from bids.tests import get_test_data_path


@pytest.fixture
def testlayout():
    """A BIDSLayout for testing."""
    data_dir = join(get_test_data_path(), "synthetic")
    return BIDSLayout(data_dir)


def test_bval_smoke(testlayout):
    """Smoke test for parsing _dwi.bval

    It should return a str description when provided valid inputs.
    """
    bval_file = testlayout.get(
        subject="01",
        session="01",
        suffix="dwi",
        extension=[".bval"],
    )

    bval_str = parameters.describe_bvals(bval_file[0])
    assert isinstance(bval_str, str)


def test_describe_echo_times_smoke(testlayout):
    """Smoke test for parsing echo time

    It should return a str description when provided valid inputs.
    """
    anat_file = testlayout.get(
        subject="01",
        session="01",
        suffix="T1w",
        extension=[".nii.gz"],
    )

    te_str, me_str = parameters.describe_echo_times(anat_file)
    assert isinstance(te_str, str)


def test_describe_echo_times_fmap(testlayout):
    """Smoke test for parsing echo time

    It should return a str description when provided valid inputs.
    """
    fmap_file = testlayout.get(
        subject="01",
        session="01",
        suffix="phasediff",
        extension=[".nii.gz"],
    )

    te_str = parameters.describe_echo_times_fmap(fmap_file)
    assert isinstance(te_str, str)


def test_describe_repetition_time_smoke():
    
    # given
    metadata = {"RepetitionTime": 2}
    # when
    tr_str = parameters.describe_repetition_time(metadata)
    # then
    expected = "repetition time, TR=2000ms"
    assert tr_str == expected


def test_describe_func_duration_smoke():
    
    # given
    n_vols = 100
    tr = 2.5
    # when
    duration = parameters.describe_func_duration(n_vols, tr)
    # then
    expected = "4:10"
    assert duration == expected

def test_describe_multiband_factor_smoke():
    
    # given
    metadata = {"MultibandAccelerationFactor": 2}
    # when
    mb_str = parameters.describe_multiband_factor(metadata)
    # then
    expected = "MB factor=2"
    assert mb_str == expected

    # given
    metadata = {"RepetitionTime": 2}
    # when
    mb_str = parameters.describe_multiband_factor(metadata)
    # then
    expected = ""
    assert mb_str == expected    

def test_describe_inplane_accel_smoke():
    
    # given
    metadata = {"ParallelReductionFactorInPlane": 2}
    # when
    mb_str = parameters.describe_inplane_accel(metadata)
    # then
    expected = "in-plane acceleration factor=2"
    assert mb_str == expected

    # given
    metadata = {"RepetitionTime": 2}
    # when
    mb_str = parameters.describe_inplane_accel(metadata)
    # then
    expected = ""
    assert mb_str == expected    

def test_describe_flip_angle_smoke():
    
    # given
    metadata = {"FlipAngle": 90}
    # when
    mb_str = parameters.describe_flip_angle(metadata)
    # then
    expected = "flip angle, FA=90<deg>"
    assert mb_str == expected