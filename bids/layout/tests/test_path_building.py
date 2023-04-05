"""Tests of path-building functionality."""

from os.path import join
from pathlib import Path

import pytest
from bids.layout import BIDSLayout
from bids.tests import get_test_data_path


@pytest.fixture(scope='module')
def layout():
    data_dir = join(get_test_data_path(), '7t_trt')
    return BIDSLayout(data_dir)


def test_bold_construction(layout):
    ents = dict(subject='01', run=1, task='rest', suffix='bold')
    relative = Path("sub-01") / "func" / "sub-01_task-rest_run-1_bold.nii.gz"
    absolute = Path(layout.root) / relative
    assert layout.build_path(ents, absolute_paths=False) == str(relative)
    assert layout.build_path(ents, absolute_paths=True) == str(absolute)
    # layout fixture created with `absolute_paths=True`, defaulting to absolute
    assert layout.build_path(ents) == str(absolute)


def test_invalid_file_construction(layout):
    # no hyphens allowed!
    ents = dict(subject='01', run=1, task='resting-state', suffix='bold')
    with pytest.raises(ValueError):
        layout.build_path(ents)

    target = "sub-01/func/sub-01_task-resting-state_run-1_bold.nii.gz"
    assert layout.build_path(ents, validate=False, absolute_paths=False) == target


def test_failed_file_construction(layout):
    ents = dict(subject='01', fakekey='foobar')
    with pytest.raises(ValueError):
        layout.build_path(ents, strict=True)


@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize("validate", [True, False])
def test_insufficient_entities(layout, strict, validate):
    """Check https://github.com/bids-standard/pybids/pull/574#discussion_r366447600."""
    with pytest.raises(ValueError):
        layout.build_path({'subject': '01'}, strict=strict, validate=validate)


"""
The following tests that indexes some datasets from bids examples,
parses the files for entities and
reconstructs the fullpath of each file by relying on pybids config
and compares it what's actually in the dataset
"""

@pytest.mark.parametrize(
    "dataset",
    [
        ("qmri_irt1"),
        ("qmri_mese"),
        ("qmri_mp2rage"),
        ("qmri_mp2rageme"),
        ("qmri_mtsat"),
        ("qmri_sa2rage"),
        ("qmri_vfa"),
        ("synthetic"),
        ("ds000001-fmriprep"),
    ],
)
def test_path_building_on_derivative_examples(dataset, bids_examples):
    layout = BIDSLayout(bids_examples / dataset, derivatives=True)
    for derivative in layout.derivatives:
        sublayout = layout.derivatives[derivative]
        files = sublayout.get(subject=".*", datatype=".*", regex_search=True)
        for bf in files:
            entities = bf.get_entities()

            # Some examples include unfinalized derivatives
            if entities["suffix"] in {"timeseries"}:  # BEP012
                continue

            path = sublayout.build_path(entities, strict=True, validate=False)
            assert path == bf.path

@pytest.mark.parametrize(
    "dataset",
    [
        ("micr_SEM"),
        ("micr_SPIM"),
        ("asl001"),
        ("asl002"),
        ("asl003"),
        ("asl004"),
        ("asl005"),
        ("pet001"),
        ("pet002"),
        ("pet003"),
        ("pet004"),
        ("pet005"),
        ("qmri_megre"),
        ("qmri_tb1tfl"),
        ("eeg_cbm"),
        ("ieeg_filtered_speech"),
        ("ieeg_visual_multimodal"),
        ("ds000248"),
        ("ds001"),
        ("ds114"),
        ("qmri_irt1"),
        ("qmri_mese"),
        ("qmri_mp2rage"),
        ("qmri_mp2rageme"),
        ("qmri_mtsat"),
        ("qmri_sa2rage"),
        ("qmri_vfa"),
        ("ds000117"),
    ],
)
def test_path_building_in_raw_scope(dataset, bids_examples):
    layout = BIDSLayout(bids_examples / dataset, derivatives=False)
    files = layout.get(subject=".*", datatype=".*", regex_search=True, scope="raw")
    for bf in files:
        entities = bf.get_entities()
        path = layout.build_path(entities)
        assert path == bf.path

@pytest.mark.parametrize("scope", ["raw"])
@pytest.mark.parametrize(
    "dataset",
    [
        pytest.param(
            "ds000247",
            marks=pytest.mark.xfail(strict=True,
                reason="meg ds folder"
            ),
        ),
        pytest.param(
            "ds000246",
            marks=pytest.mark.xfail(strict=True,
                reason="meg ds folder"
            ),
        ),
    ],
)
def test_path_building_on_examples_with_derivatives_meg_ds_folder(dataset, scope, bids_examples):
    layout = BIDSLayout(bids_examples / dataset, derivatives=True)
    files = layout.get(subject=".*", datatype=".*", regex_search = True, scope=scope)
    for bf in files:
        entities = bf.get_entities()
        path = layout.build_path(entities)
        assert(path==bf.path)
