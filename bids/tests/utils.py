""" Test-related utilities """

from pathlib import Path


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
