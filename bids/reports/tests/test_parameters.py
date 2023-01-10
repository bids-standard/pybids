import json
from os.path import abspath, join

import nibabel as nib
import pytest

from bids.layout import BIDSLayout
from bids.reports import parameters
from bids.tests import get_test_data_path


@pytest.fixture
def testlayout():
    """A BIDSLayout for testing."""
    data_dir = join(get_test_data_path(), "synthetic")
    return BIDSLayout(data_dir)


@pytest.fixture
def testimg(testlayout):

    func_files = testlayout.get(
        subject="01",
        session="01",
        task="nback",
        run="01",
        extension=[".nii.gz"],
    )
    return nib.load(func_files[0].path)


@pytest.fixture
def testdiffimg(testlayout):

    dwi_files = testlayout.get(
        subject="01",
        session="01",
        datatype="dwi",
        extension=[".nii.gz"],
    )
    return nib.load(dwi_files[0].path)


@pytest.fixture
def testconfig():
    config_file = abspath(
        join(get_test_data_path(), "../../reports/config/converters.json")
    )
    with open(config_file, "r") as fobj:
        config = json.load(fobj)
    return config


@pytest.fixture
def testmeta():
    return {
        "RepetitionTime": 2.0,
        "MultibandAccelerationFactor": 2,
        "ParallelReductionFactorInPlane": 2,
        "FlipAngle": 90,
        "PhaseEncodingDirection": "i",
        "SliceTiming": [0, 1, 2, 3],
    }


@pytest.fixture
def testmeta_light():
    return {"RepetitionTime": 2.0}


@pytest.mark.parametrize(
    "ScanningSequence, expected_seq",
    [
        ("EP", "echo planar (EP)"),
        ("GR", "gradient recalled (GR)"),
        ("IR", "inversion recovery (IR)"),
        ("RM", "research mode (RM)"),
        ("SE", "spin echo (SE)"),
        ("SE_EP", "spin echo and echo planar (SE/EP)"),
        ("spam egg", "UNKNOwN SEQUENCE"),
    ],
)
def test_sequence(ScanningSequence, expected_seq, testconfig):
    """test for sequence and variant type description"""
    metadata = {
        "ScanningSequence": ScanningSequence,
    }
    seqs = parameters.sequence(metadata, testconfig)
    assert seqs == expected_seq


@pytest.mark.parametrize(
    "SequenceVariant, expected_var",
    [
        ("MP", "MAG prepared"),
        ("MTC", "magnetization transfer contrast"),
        ("NONE", "no sequence variant"),
        ("OSP", "oversampling phase"),
        ("SK", "segmented k-space"),
        ("SS", "steady state"),
        ("TRSS", "time reversed steady state"),
        ("MP_SS", "MAG prepared and steady state"),
        ("spam", "UNKNOwN SEQUENCE VARIANT"),
    ],
)
def test_variants(SequenceVariant, expected_var, testconfig):
    """test for sequence and variant type description"""
    metadata = {
        "SequenceVariant": SequenceVariant,
    }
    variants = parameters.variants(metadata, testconfig)
    assert variants == expected_var


def test_bvals_smoke(testlayout):
    """Smoke test for parsing _dwi.bval

    It should return a str description when provided valid inputs.
    """
    bval_file = testlayout.get(
        subject="01",
        session="01",
        suffix="dwi",
        extension=[".bval"],
    )

    bval_str = parameters.bvals(bval_file[0])
    assert isinstance(bval_str, str)


def test_echo_times_fmap(testlayout):
    """Smoke test for parsing echo time

    It should return a str description when provided valid inputs.
    """
    fmap_file = testlayout.get(
        subject="01",
        session="01",
        suffix="phasediff",
        extension=[".nii.gz"],
    )

    te_1, te_2 = parameters.echo_times_fmap(fmap_file)
    assert isinstance(te_1, float)


def test_describe_func_duration_smoke():

    # given
    n_vols = 100
    tr = 2.5
    # when
    duration = parameters.func_duration(n_vols, tr)
    # then
    expected = "4:10"
    assert duration == expected


def test_multiband_factor_smoke(testmeta, testmeta_light):

    # when
    multiband_factor = parameters.multiband_factor(testmeta)
    # then
    expected = "MB factor=2"
    assert multiband_factor == expected

    # when
    multiband_factor = parameters.multiband_factor(testmeta_light)
    # then
    expected = ""
    assert multiband_factor == expected


def test_inplane_accel_smoke(testmeta, testmeta_light):

    # when
    multiband_factor = parameters.inplane_accel(testmeta)
    # then
    expected = "in-plane acceleration factor=2"
    assert multiband_factor == expected

    # when
    multiband_factor = parameters.inplane_accel(testmeta_light)
    # then
    expected = ""
    assert multiband_factor == expected


@pytest.mark.parametrize(
    "slice_times, expected",
    [
        ((1, 2, 3, 4), "sequential ascending"),
        ([4, 3, 2, 1], "sequential descending"),
        ([1, 3, 2, 4], "interleaved ascending"),
        ([4, 2, 3, 1], "interleaved descending"),
    ],
)
def test_get_slice_info(slice_times, expected):

    slice_order_name = parameters.get_slice_info(slice_times)
    assert slice_order_name == expected


def test_slice_timing(testmeta):

    slice_str = parameters.slice_order(testmeta)
    expected = " in sequential ascending order"
    assert slice_str == expected


def test_intendedfor_targets(testmeta_light, testlayout):

    intended_for = parameters.intendedfor_targets(testmeta_light, testlayout)
    assert intended_for == ""

    fmap_files = testlayout.get(
        subject="01",
        session="01",
        suffix="phasediff",
        extension=[".nii.gz"],
    )
    metadata = fmap_files[0].get_metadata()
    intended_for = parameters.intendedfor_targets(metadata, testlayout)
    assert intended_for == " for the first and second runs of the N-Back BOLD scan"
