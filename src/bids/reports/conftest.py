import json
import warnings
from importlib.resources import files

import pytest

from bids.layout import BIDSLayout


@pytest.fixture(scope='session')
def testlayout(tests_dir):
    """A BIDSLayout for testing."""
    return BIDSLayout(tests_dir / 'data' / 'synthetic')


@pytest.fixture(scope='session')
def config_file():
    # PY39: config isn't a module, so can't be used in files()
    # This is relaxed in more recent Pythons
    return files('bids.reports') / 'config' / 'converters.json'


@pytest.fixture(scope='session')
def testconfig(config_file):
    return json.loads(config_file.read_text())


@pytest.fixture(autouse=True)
def _unsupported_files():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r'.*magnitude.* not yet supported.*',
            category=UserWarning,
            append=True,
        )
        yield
