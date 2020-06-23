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
