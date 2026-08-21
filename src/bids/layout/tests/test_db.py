"""Test functionality in the db module--mostly related to connection
management."""

from bids.layout import BIDSLayout
from bids.layout.db import get_database_file


def test_get_database_file(tmp_path):
    assert get_database_file(None) is None
    new_path = tmp_path / "a_new_subdir"
    assert not new_path.exists()
    db_file = get_database_file(new_path)
    assert db_file == new_path / 'layout_index.sqlite'
    assert new_path.exists()


def test_save_database_twice_same_path(tests_dir, tmp_path):
    """Saving to the same path twice must be idempotent (gh-1079)."""
    layout = BIDSLayout(tests_dir / 'data' / 'ds005')
    db_path = str(tmp_path / 'db')

    layout.save(db_path)
    # This used to raise sqlite3.OperationalError: table associations
    # already exists, because the destination already held the schema.
    layout.save(db_path)

    # The re-saved database must still be complete and reloadable.
    reloaded = BIDSLayout(tests_dir / 'data' / 'ds005', database_path=db_path)
    assert sorted(layout.get(return_type='file')) == sorted(
        reloaded.get(return_type='file')
    )
