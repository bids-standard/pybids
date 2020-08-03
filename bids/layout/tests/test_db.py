"""Test functionality in the db module--mostly related to connection
management."""

import re
from pathlib import Path

from bids.layout.db import (ConnectionManager, get_database_file,
                            get_database_sidecar, _sanitize_init_args)


def test_sanitize_init_args():
    patt = [re.compile('f1ct.*n'), 'code']
    root = Path('.') / 'fictional_path'
    result = _sanitize_init_args(ignore=patt, root=root, absolute_paths=False)
    assert result['absolute_paths'] == False
    assert isinstance(result['ignore'], list)
    assert all([isinstance(el, str) for el in result['ignore']])
    assert isinstance(result['root'], str)


def test_get_database_file(tmp_path):
    tmp_path = Path(str(tmp_path))  # PY35: pytest uses pathlib2.Path; __fspath__ in Python 3.6 fixes this
    assert get_database_file(None) is None
    new_path = tmp_path / "a_new_subdir"
    assert not new_path.exists()
    db_file = get_database_file(new_path)
    assert db_file == new_path / 'layout_index.sqlite'
    assert new_path.exists()


def test_get_database_sidecar():
    db_file = '/abs/path/to/db/file.sqlite'
    f1 = get_database_sidecar(db_file)
    assert isinstance(f1, Path)
    assert str(f1) == '/abs/path/to/db/layout_args.json'
    f2 = get_database_sidecar(Path(db_file))
    assert f1 == f2
