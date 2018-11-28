import sys
import pytest
from pathlib import Path
from bids.layout import BIDSLayout
from bids.tests import get_test_data_path
"""
test handling of pathlib Path file paths in place of old-style string type
"""

TESTPATH = Path(get_test_data_path()).joinpath("ds005")
TESTSTR = str(TESTPATH)
FALSEPATH = TESTPATH.joinpath("junk")
FALSESTR = str(FALSEPATH)
assert not FALSEPATH.exists()

def test_strroot_pos():
    layout = BIDSLayout(TESTSTR)

def test_strroot_neg():
    with pytest.raises(ValueError):
        layout = BIDSLayout(FALSESTR)

@pytest.mark.skipif(sys.version_info < (3,4),
        reason="requires python3.4 or higher")
def test_pathroot_pos():
    layout = BIDSLayout(TESTPATH)

@pytest.mark.skipif(sys.version_info < (3,4),
        reason="requires python3.4 or higher")
def test_pathroot_neg():
    with pytest.raises(ValueError):
        layout = BIDSLayout(FALSEPATH)

@pytest.mark.skipif(sys.version_info >= (3,4), reason="test of exception handling for older pythons")
def test_pathroot_pos_oldpython():
    with pytest.raises(TypeError):
        layout = BIDSLayout(TESTPATH)

@pytest.mark.skipif(sys.version_info >= (3,4), reason="test of exception handling for older pythons")
def test_pathroot_neg_oldpython():
    with pytest.raises(TypeError):
        layout = BIDSLayout(FALSEPATH)
