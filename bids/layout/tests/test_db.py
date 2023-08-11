"""Test functionality in the db module--mostly related to connection
management."""

from bids.layout.db import get_database_file


def test_get_database_file(tmp_path):
    assert get_database_file(None) is None
    new_path = tmp_path / "a_new_subdir"
    assert not new_path.exists()
    db_file = get_database_file(new_path)
    assert db_file == new_path / 'layout_index.sqlite'
    assert new_path.exists()
