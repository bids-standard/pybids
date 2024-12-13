"""
This module allows you to mock the config file as needed.
A default fixture that simply returns a safe-to-modify copy of
the default value is provided.
This can be overridden by parametrizing over the option you wish to
mock.

e.g.

>>> @pytest.mark.parametrize("extension_initial_dot", (True, False))
... def test_fixture(mock_config, extension_initial_dot):
...     import bids
...     assert bids.config.get_option("extension_initial_dot") == extension_initial_dot
"""

import os
from upath import UPath as Path
from unittest.mock import patch

import pytest

@pytest.fixture
def config_paths():
    import bids.config
    return bids.config.get_option('config_paths').copy()

@pytest.fixture
def extension_initial_dot():
    import bids.config
    return bids.config.get_option('extension_initial_dot')

@pytest.fixture
def mock_config(config_paths, extension_initial_dot):
    import bids.config
    with patch.dict('bids.config._settings'):
        bids.config._settings['config_paths'] = config_paths
        bids.config._settings['extension_initial_dot'] = extension_initial_dot
        yield

@pytest.fixture(scope='session')
def bids_examples():
    examples_dir = Path(os.getenv(
        "BIDS_EXAMPLES",
        Path(__file__).absolute().parent.parent.parent / "bids-examples"
    ))

    if not Path.is_dir(examples_dir / "ds001"):
        pytest.skip(
            f"BIDS examples missing from {examples_dir}. "
            "Override default location with BIDS_EXAMPLES environment variable."
        )
    return examples_dir

@pytest.fixture(scope='session')
def tests_dir():
    test_dir = Path(os.getenv(
        "PYBIDS_TEST_DATA",
        Path(__file__).absolute().parent.parent.parent / "tests"
    ))

    if not Path.is_dir(test_dir / "data"):
        pytest.skip(
            f"BIDS examples missing from {test_dir}. "
            "Override default location with PYBIDS_TEST_DATA environment variable."
        )
    return test_dir
