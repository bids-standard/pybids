import pytest

from os.path import abspath, join

from bids.layout import BIDSLayout
from bids.reports import parameters
from bids.tests import get_test_data_path


@pytest.fixture
def testlayout():
    """A BIDSLayout for testing."""
    data_dir = join(get_test_data_path(), "synthetic")
    return BIDSLayout(data_dir)


def test_bval_smoke(testlayout):
    """Smoke test for parsing.anat_info.

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
