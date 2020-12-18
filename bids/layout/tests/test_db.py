"""Test functionality in the db module--mostly related to connection
management."""

import re
from pathlib import Path

from bids.layout.db import (ConnectionManager, get_database_file)


def test_get_database_file(tmp_path):
    tmp_path = Path(str(tmp_path))  # PY35: pytest uses pathlib2.Path; __fspath__ in Python 3.6 fixes this
    assert get_database_file(None) is None
    new_path = tmp_path / "a_new_subdir"
    assert not new_path.exists()
    db_file = get_database_file(new_path)
    assert db_file == new_path / 'layout_index.sqlite'
    assert new_path.exists()
