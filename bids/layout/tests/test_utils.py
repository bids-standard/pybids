""" Test-related utilities """

from ..utils import write_derivative_description
from ...tests import get_test_data_path
from .. import BIDSLayout


def test_write_derivative_description(exist_ok=True):
    """Test write_derivative_description(source_dir, name, bids_version='1.1.1', **desc_kwargs). """

    source_dir = get_test_data_path("Path") / '7t_trt'
    write_derivative_description(source_dir, name="test", bids_version='1.1.1', exist_ok=exist_ok)
    BIDSLayout(source_dir, derivatives=True)
