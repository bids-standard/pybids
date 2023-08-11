"""
Database-related functionality.
"""

from pathlib import Path
import re
import sqlite3
from functools import lru_cache

import sqlalchemy as sa

from bids.utils import listify
from .models import Base, Config, LayoutInfo


def get_database_file(path):
    if path is not None:
        path = Path(path)
        database_file = path / 'layout_index.sqlite'
        path.mkdir(exist_ok=True, parents=True)
    else:
        database_file = None
    return database_file


class ConnectionManager:

    def __init__(self, database_path=None, reset_database=False, config=None,
                 init_args=None):

        self.database_file = get_database_file(database_path)

        # Determine if file exists before we create it in _get_engine()
        reset_database = (
            reset_database or                # manual reset
            self.database_file is None or    # in-memory DB
            not self.database_file.exists()  # file hasn't been created yet
        )

        self.engine = self._get_engine(self.database_file)
        self.sessionmaker = sa.orm.sessionmaker(bind=self.engine)
        self._session = None

        if reset_database:
            self.reset_database(init_args, config)

        self._database_reset = reset_database

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
            if not isinstance(item, str):
                return False
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

    @classmethod
    def exists(cls, database_path):
        return get_database_file(database_path).exists()

    def reset_database(self, init_args, config):
        Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)
        # Add LayoutInfo record
        if not isinstance(init_args, LayoutInfo):
            layout_info = LayoutInfo(**init_args)
        self.session.add(layout_info)
        # Add config records
        config = [Config.load(c, session=self.session) for c in listify(config)]
        self.session.add_all(config)
        self.session.commit()

    def save_database(self, database_path, replace_connection=True):
        """Save the current index as a SQLite3 DB at the specified location.

        Note: This is only necessary if a database_path was not specified
        at initialization, and the user now wants to save the index.
        If a database_path was specified originally, there is no need to
        re-save using this method.

        Parameters
        ----------
        database_path : str
            The path to the desired database folder. By default,
            uses .db_cache. If a relative path is passed, it is assumed to
            be relative to the BIDSLayout root directory.
        replace_connection : bool, optional
            If True, returns a new ConnectionManager that points to the newly
            created database. If False, returns the current instance.
        """
        database_file = get_database_file(database_path)
        new_db = sqlite3.connect(str(database_file))
        old_db = self.engine.connect().connection

        with new_db:
            for line in old_db.iterdump():
                if line not in ('BEGIN;', 'COMMIT;'):
                    new_db.execute(line)
            new_db.commit()

        if replace_connection:
            return ConnectionManager(database_path, init_args=self.layout_info)
        else:
            return self

    @property
    def session(self):
        if self._session is None:
            self.reset_session()
        return self._session

    @property
    @lru_cache()
    def layout_info(self):
        return self.session.query(LayoutInfo).first()


    def reset_session(self):
        """Force a new session."""
        self._session = self.sessionmaker()
