"""
Test handling of pathlib Path file paths in place of old-style string type.
"""

import sys
import pytest
from pathlib import Path

from bids.layout import BIDSLayout
from bids.tests import get_test_data_path


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

def test_pathroot_pos():
    layout = BIDSLayout(TESTPATH)

def test_pathroot_neg():
    with pytest.raises(ValueError):
        layout = BIDSLayout(FALSEPATH)
