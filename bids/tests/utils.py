""" Test-related utilities """

from pathlib import Path
from ..utils import write_derivative_description
from .. import BIDSLayout

def get_test_data_path(return_type="str"):
    """
    :param return_type: Specify the type of object returned. Can be 'str'
                        (default, for backward-compatibility) or 'Path' for pathlib.Path type.
    :return: The path for testing data.
    """

    path = Path(__file__).resolve().parent / 'data'
    if return_type == "str":
        return str(path)
    elif return_type == "Path":
        return path
    else:
        raise ValueError("return_type can be 'str' or 'Path. Got {}.".format(return_type))


def test_write_derivative_description(exist_ok=True):
    """Test write_derivative_description(source_dir, name, bids_version='1.1.1', **desc_kwargs). """

    source_dir = get_test_data_path("Path") / '7t_trt'
    write_derivative_description(source_dir, name="test", bids_version='1.1.1', exist_ok=exist_ok)
    BIDSLayout(source_dir, derivatives=True)
