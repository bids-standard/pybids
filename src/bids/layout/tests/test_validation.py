"""Tests of BIDSValidator functionality."""

from os.path import join

import pytest

from bids_validator import BIDSValidator
from bids.layout import BIDSLayout
from bids.tests import get_test_data_path


# Fixture uses in the rest of the tests
@pytest.fixture
def testvalidator():
    return BIDSValidator()


def test_layout_with_validation():
    data_dir = join(get_test_data_path(), '7t_trt')
    layout1 = BIDSLayout(data_dir, validate=True)
    layout2 = BIDSLayout(data_dir, validate=False)
    assert len(layout1.files) < len(layout2.files)
    # Not a valid BIDS file
    badfile = join(data_dir, 'test.bval')
    assert badfile not in layout1.files
    assert badfile in layout2.files
