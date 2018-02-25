from os.path import join
from bids.grabbids import BIDSLayout
from bids.reports import BIDSReport
from bids.tests import get_test_data_path
import pytest


@pytest.fixture
def testlayout():
    data_dir = join(get_test_data_path(), 'synthetic')
    return BIDSLayout(data_dir)


def test_report_init():
    report = BIDSReport(testlayout)
    assert isinstance(report, BIDSReport)
