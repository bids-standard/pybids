""" Test-related utilities """
from pathlib import Path
import pkg_resources


def get_test_data_path(return_type="str"):
    """
    Parameters
    ----------
    return_type : str or Path
        Specify the type of object returned. Can be 'str'
        (default, for backward-compatibility) or 'Path' for pathlib.Path type.

    Returns
    -------
    The path for testing data.
    """

    path = pkg_resources.resource_filename('bids', 'tests/data')
    if return_type == "str":
        return path
    elif return_type == "Path":
        return Path(path)
    else:
        raise ValueError("return_type can be 'str' or 'Path. Got {}.".format(return_type))
