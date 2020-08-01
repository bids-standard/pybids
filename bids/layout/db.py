"""
Database-related functionality.
"""

from pathlib import Path
import json
import re
import warnings
from functools import lru_cache

import sqlalchemy as sa
from sqlalchemy.orm import joinedload

from bids.utils import listify
from .models import Base, Config


def _make_db_path(path):
    if path is not None:
        path = Path(path)
        database_file = path / 'layout_index.sqlite'
        path.mkdir(exist_ok=True, parents=True)
    else:
        database_file = None
    return database_file

def get_database_sidecar(path):
    """Given a path to a database file, return the associated sidecar.

    Args:
        path (str, Path): A path to a database file
    """
    if isinstance(path, str):
        path = Path(path)
    return path.parent() / 'layout_args.json'


class ConnectionManager:

    def __init__(self, database_path=None, reset_database=False, config=None,
                 init_args=None):

        self.database_file = _make_db_path(database_path)
        self.engine = self._get_engine(self.database_file)
        self.sessionmaker = sa.orm.sessionmaker(bind=self.engine)
        self._session = None

        # Determine whether to reset DB or load from file
        reset_database = (
            reset_database or  # Manual Request
            not self.database_file or  # In memory transient db
            not self.database_file.exists()  # New file-based db created
        )

        if reset_database:
            self.reset_database(init_args, config)
        else:
            self.load_database(init_args)

    def _get_engine(self, database_file):
        if database_file is not None:
            # https://docs.sqlalchemy.org/en/13/dialects/sqlite.html
            # When a file-based database is specified, the dialect will use
            # NullPool as the source of connections. This pool closes and
            # discards connections which are returned to the pool immediately.
            # SQLite file-based connections have extremely low overhead, so
            # pooling is not necessary. The scheme also prevents a connection
            # from being used again in a different thread and works best
            # with SQLite's coarse-grained file locking.
            from sqlalchemy.pool import NullPool
            engine = sa.create_engine(
                'sqlite:///{dbfilepath}'.format(dbfilepath=database_file),
                connect_args={'check_same_thread': False},
                poolclass=NullPool)
        else:
            # https://docs.sqlalchemy.org/en/13/dialects/sqlite.html
            # Using a Memory Database in Multiple Threads
            # To use a :memory: database in a multithreaded scenario, the same
            # connection object must be shared among threads, since the
            # database exists only within the scope of that connection. The
            # StaticPool implementation will maintain a single connection
            # globally, and the check_same_thread flag can be passed to
            # Pysqlite as False. Note that using a :memory: database in
            # multiple threads requires a recent version of SQLite.
            from sqlalchemy.pool import StaticPool
            engine = sa.create_engine(
                'sqlite://',  # In memory database
                connect_args={'check_same_thread': False},
                poolclass=StaticPool)
            
        def regexp(expr, item):
            """Regex function for SQLite's REGEXP."""
            reg = re.compile(expr, re.I)
            return reg.search(item) is not None

        engine.connect()

        # Do not remove this decorator!!! An in-line create_function call will
        # work when using an in-memory SQLite DB, but fails when using a file.
        # For more details, see https://stackoverflow.com/questions/12461814/
        @sa.event.listens_for(engine, "begin")
        def do_begin(conn):
            conn.connection.create_function('regexp', 2, regexp)

        return engine

    def reset_database(self, init_args=None, config=None):
        init_args = init_args or {}
        Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)
        if self.database_sidecar is not None:
            self.database_sidecar.write_text(json.dumps(init_args))
        # Add config records
        config = listify('bids' if config is None else config)
        config = [Config.load(c, session=self.session) for c in listify(config)]
        self.session.add(*config)
        self.session.commit()

    def load_database(self, init_args=None):
        if self.database_file is None:
            raise ValueError("load_database() can only be called on databases "
                             "stored in a file, not on in-memory databases.")
        init_args = init_args or {}
        saved_args = json.loads(self.database_sidecar.read_text())
        for k, v in saved_args.items():
            if init_args[k] != v:
                raise ValueError(
                    "Initialization argument ('{}') does not match "
                    "for database_path: {}.\n"
                    "Saved value: {}.\n"
                    "Current value: {}.".format(
                        k, self.database_file, v, init_args[k])
                    )  

    @property
    # Replace with @cached_property (3.8+) at some point in future
    @lru_cache(maxsize=None)
    def database_sidecar(self):
        if self.database_file is not None:
            return get_database_sidecar(self.database_file)
        return None

    @property
    def session(self):
        if self._session is None:
            self.reset_session()
        return self._session

    def reset_session(self):
        """Force a new session."""
        self._session = self.sessionmaker()