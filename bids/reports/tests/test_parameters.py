import pytest
import json
import nibabel as nib

from os.path import abspath, join

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
    ],
)
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
    ],
)
def test_describe_sequence(
    ScanningSequence, expected_seq, SequenceVariant, expected_var, testconfig
):
    """test for sequence and variant type description"""
    metadata = {
        "SequenceVariant": SequenceVariant,
        "ScanningSequence": ScanningSequence,
    }
    seqs, variants = parameters.describe_sequence(metadata, testconfig)

    assert seqs == expected_seq
    assert variants == expected_var


@pytest.mark.parametrize(
    "pe_direction, expected",
    [
        ("i", "left to right"),
        ("i-", "right to left"),
        ("j", "posterior to anterior"),
        ("j-", "anterior to posterior"),
        ("k", "inferior to superior"),
        ("k-", "superior to inferior"),
    ],
)
def test_describe_pe_direction(pe_direction, expected, testconfig):
    """test for phase encoding direction description"""
    metadata = {"PhaseEncodingDirection": pe_direction}
    dir_str = parameters.describe_pe_direction(metadata, testconfig)
    assert dir_str == "phase encoding: " + expected


def test_describe_bvals_smoke(testlayout):
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


def test_describe_repetition_time_smoke(testmeta):

    # when
    tr_str = parameters.describe_repetition_time(testmeta)
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


def test_describe_multiband_factor_smoke(testmeta, testmeta_light):

    # when
    mb_str = parameters.describe_multiband_factor(testmeta)
    # then
    expected = "MB factor=2"
    assert mb_str == expected

    # when
    mb_str = parameters.describe_multiband_factor(testmeta_light)
    # then
    expected = ""
    assert mb_str == expected


def test_describe_inplane_accel_smoke(testmeta, testmeta_light):

    # when
    mb_str = parameters.describe_inplane_accel(testmeta)
    # then
    expected = "in-plane acceleration factor=2"
    assert mb_str == expected

    # when
    mb_str = parameters.describe_inplane_accel(testmeta_light)
    # then
    expected = ""
    assert mb_str == expected


def test_describe_flip_angle_smoke(testmeta):

    # when
    mb_str = parameters.describe_flip_angle(testmeta)
    # then
    expected = "flip angle, FA=90<deg>"
    assert mb_str == expected


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


def test_describe_slice_timing(testimg, testmeta, testmeta_light):

    slice_str = parameters.describe_slice_timing(testimg, testmeta)
    expected = "4 slices in sequential ascending order"
    assert slice_str == expected

    slice_str = parameters.describe_slice_timing(testimg, testmeta_light)
    expected = "64 slices"
    assert slice_str == expected


def test_get_size_str(testimg):

    voxel_size, matrix_size, fov = parameters.get_size_str(testimg)
    expected_vox = "2x2x2"
    expected_mat = "64x64"
    expected_fov = "128x128"
    assert voxel_size == expected_vox
    assert matrix_size == expected_mat
    assert fov == expected_fov


def test_describe_image_size(testimg):

    fov_str, matrixsize_str, voxelsize_str = parameters.describe_image_size(testimg)
    expected_vox = "voxel size=2x2x2mm"
    expected_mat = "matrix size=64x64"
    expected_fov = "field of view, FOV=128x128mm"
    assert voxelsize_str == expected_vox
    assert matrixsize_str == expected_mat
    assert fov_str == expected_fov


def test_describe_dmri_directions(testdiffimg):

    dif_dir_str = parameters.describe_dmri_directions(testdiffimg)
    expected = "64 diffusion directions"
    assert dif_dir_str == expected


def test_describe_intendedfor_targets(testmeta_light, testlayout):

    for_str = parameters.describe_intendedfor_targets(testmeta_light, testlayout)
    assert for_str == ""

    fmap_files = testlayout.get(
        subject="01",
        session="01",
        suffix="phasediff",
        extension=[".nii.gz"],
    )
    metadata = fmap_files[0].get_metadata()
    for_str = parameters.describe_intendedfor_targets(metadata, testlayout)
    assert for_str == " for the first and second runs of the N-Back BOLD scan"
