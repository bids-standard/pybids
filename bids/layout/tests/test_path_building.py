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
