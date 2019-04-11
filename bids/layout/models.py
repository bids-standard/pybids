from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.orm.collections import attribute_mapped_collection
from sqlalchemy import (Column, Integer, String, Boolean, ForeignKey, JSON,
                        Table)
from sqlalchemy.orm import reconstructor, relationship, backref, object_session
import re
import os
import warnings
import json
from copy import deepcopy

from .writing import build_path, write_contents_to_file
from ..utils import listify, check_path_matches_patterns
from ..config import get_option
from ..external import six


Base = declarative_base()


class Config(Base):
    """ Container for BIDS configuration information.

    Args:
        name (str): The name to give the Config (e.g., 'bids').
        entities (list): A list of dictionaries containing entity configuration
            information.
        default_path_patterns (list): Optional list of patterns used to build
            new paths.
    """
    __tablename__ = 'configs'

    name = Column(String, primary_key=True)
    _default_path_patterns = Column(JSON)
    entities = relationship("Entity", secondary="config_to_entity_map",
                            collection_class=attribute_mapped_collection('name'))

    def __init__(self, name, entities=None, default_path_patterns=None,
                 session=None):

        self.name = name
        self.default_path_patterns = default_path_patterns
        self._default_path_patterns = json.dumps(default_path_patterns)

        if entities:
            for ent in entities:
                if session is not None:
                    existing = session.query(Config).filter_by(name=ent['name']).first()
                else:
                    existing = None
                ent = existing or Entity(**ent)
                self.entities[ent.name] = ent
                if session is not None:
                    session.add_all(list(self.entities.values()))
                    session.commit()

    @reconstructor
    def _init_on_load(self):
        self.default_path_patterns = json.loads(self._default_path_patterns)

    @classmethod
    def load(self, config, session=None):

        if isinstance(config, six.string_types):
            config_paths = get_option('config_paths')
            if config in config_paths:
                config = config_paths[config]
            if not os.path.exists(config):
                raise ValueError("{} is not a valid path.".format(config))
            else:
                with open(config, 'r') as f:
                    config = json.load(f)

        # Return existing Config record if one exists
        if session is not None:
            result = session.query(Config).filter_by(name=config['name']).first()
            if result:
                return result

        return Config(session=session, **config)


class BIDSFile(Base):
    __tablename__ = 'files'

    path = Column(String, primary_key=True)
    filename = Column(String)
    dirname = Column(String)
    entities = association_proxy("tags", "value")
    is_dir = Column(Boolean)

    def __init__(self, filename, derivatives=False, is_dir=False):
        self.path = filename
        self.filename = os.path.basename(self.path)
        self.dirname = os.path.dirname(self.path)
        self.derivatives = derivatives
        self.is_dir = is_dir
        self._init_on_load()

    @reconstructor
    def _init_on_load(self):
        self._data = None

    def _matches(self, entities=None, regex_search=False):
        """
        Checks whether the file matches all of the passed entities.

        Args:
            entities (dict): A dictionary of entity names -> values to match.
            regex_search (bool): Whether to require exact match (False) or
                regex search (True) when comparing the query string to each
                entity.
        Returns:
            True if _all_ entities match; False otherwise.
        """

        if entities is None:
            return True

        for name, val in entities.items():

            if (name not in self.entities) ^ (val is None):
                return False

            if val is None:
                continue

            def make_patt(x):
                patt = str(x)
                if not regex_search:
                    patt = re.escape(patt)
                if isinstance(x, (int, float)):
                    # allow for leading zeros if a number was specified
                    # regardless of regex_search
                    patt = '0*' + patt
                if not regex_search:
                    patt = '^{}$'.format(patt)
                return patt

            ent_patts = [make_patt(x) for x in listify(val)]
            patt = '|'.join(ent_patts)

            if re.search(patt, str(self.entities[name])) is None:
                return False

        return True

    def copy(self, path_patterns, symbolic_link=False, root=None,
             conflicts='fail'):
        """ Copy the contents of a file to a new location.
        
        Args:
            path_patterns (list): List of patterns use to construct the new
                filename. See build_path documentation for details.
            symbolic_link (bool): If True, use a symbolic link to point to the
                existing file. If False, creates a new file.
            root (str): Optional path to prepend to the constructed filename.
            conflicts (str): Defines the desired action when the output path
                already exists. Must be one of:
                    'fail': raises an exception
                    'skip' does nothing
                    'overwrite': overwrites the existing file
                    'append': adds  a suffix to each file copy, starting with 1
        """
        new_filename = build_path(self.entities, path_patterns)
        if not new_filename:
            return None

        if new_filename[-1] == os.sep:
            new_filename += self.filename

        if os.path.isabs(self.path) or root is None:
            path = self.path
        else:
            path = os.path.join(root, self.path)

        if not os.path.exists(path):
            raise ValueError("Target filename to copy/symlink (%s) doesn't "
                             "exist." % path)

        if symbolic_link:
            contents = None
            link_to = path
        else:
            with open(path, 'r') as f:
                contents = f.read()
            link_to = None

        write_contents_to_file(new_filename, contents=contents,
                               link_to=link_to, content_mode='text', root=root,
                               conflicts=conflicts)

    def __getattr__(self, attr):
        # Ensures backwards compatibility with old File_ namedtuple, which is
        # deprecated as of 0.7.
        # _ check first to not mask away access to __setstate__ etc.
        # AFAIK None of the entities are allowed to start with _ anyways
        # so the check is more generic than __
        if not attr.startswith('_') and attr in self.entities:
            warnings.warn("Accessing entities as attributes is deprecated as "
                          "of 0.7. Please use the .entities dictionary instead"
                          " (i.e., .entities['%s'] instead of .%s."
                          % (attr, attr))
            return self.entities[attr]
        raise AttributeError("%s object has no attribute named %r" %
                             (self.__class__.__name__, attr))

    def __repr__(self):
        return "<BIDSFile filename='{}'>".format(self.path)

    def get_df(self, include_timing=True, adjust_onset=False):
        """ Returns the contents of the file as a pandas DataFrame. """

        ext = self.entities['extension']
        if ext not in ['tsv', 'tsv.gz']:
            raise ValueError(
                "get_df() can only be called on BIDSFiles with a .tsv or "
                ".tsv.gz extension. Invalid extension: '{}'".format(ext))

        import pandas as pd
        import numpy as np

        # TODO: memoize method instead of just caching the raw data
        if self._data is None:
            self._data = pd.read_csv(self.path, sep='\t', na_values='n/a')

        data = self._data.copy()
        md = self.get_metadata()

        if ext == 'tsv.gz':
            # We could potentially include some validation here, but that seems
            # like a job for the BIDS Validator.
            data.columns = md['Columns']
            if include_timing:
                onsets = np.arange(len(data)) / md['SamplingFrequency']
                if adjust_onset:
                    onsets += md['StartTime']
                data.insert(0, 'onset', onsets)

        return data

    def get_image(self):
        """ Return the associated image file (if it exists) as a NiBabel object
        """
        try:
            import nibabel as nb
            return nb.load(self.path)
        except Exception:
            raise ValueError("'{}' does not appear to be an image format "
                             "NiBabel can read.".format(self.path))

    def get_metadata(self):
        """ Returns all metadata associated with the current file. """
        session = object_session(self)
        query = (session.query(Tag)
                        .filter_by(file_path=self.path)
                        .join(Entity)
                        .filter(Entity.is_metadata))
        return {t.entity_name: t.value for t in query.all()}


class Entity(Base):
    __tablename__ = 'entities'

    name = Column(String, primary_key=True)
    mandatory = Column(Boolean, default=False)
    pattern = Column(String)
    directory = Column(String, nullable=True)
    is_metadata = Column(Boolean, default=False)
    _dtype = Column(String, default='str')
    files = association_proxy("tags", "value")

    def __init__(self, name, pattern=None, mandatory=False, directory=None,
                 dtype='str', is_metadata=False):
        """
        Represents a single entity defined in the JSON config.

        Args:
            name (str): The name of the entity (e.g., 'subject', 'run', etc.)
            pattern (str): A regex pattern used to match against file names.
                Must define at least one group, and only the first group is
                kept as the match.
            mandatory (bool): If True, every File _must_ match this entity.
            directory (str): Optional pattern defining a directory associated
                with the entity.
            dtype (str): The optional data type of the Entity values. Must be
                one of 'int', 'float', 'bool', or 'str'. If None, no type
                enforcement will be attempted, which means the dtype of the
                value may be unpredictable.
            is_metadata (bool): Indicates whether or not the Entity is derived
                from JSON sidecars (True) or is a predefined Entity from a
                config (False).
        """
        self.name = name
        self.pattern = pattern
        self.mandatory = mandatory
        self.directory = directory
        self.is_metadata = is_metadata

        if not isinstance(dtype, six.string_types):
            dtype = dtype.__name__
        self._dtype = dtype

        self._init_on_load()

    @reconstructor
    def _init_on_load(self):
        if self._dtype not in ('str', 'float', 'int', 'bool'):
            raise ValueError("Invalid dtype '{}'. Must be one of 'int', "
                             "'float', 'bool', or 'str'.".format(self._dtype))
        self.dtype = eval(self._dtype)
        self.regex = re.compile(self.pattern) if self.pattern is not None else None

    def __iter__(self):
        for i in self.unique():
            yield(i)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        for k, v in self.__dict__.items():
            new_val = getattr(self, k) if k == 'regex' else deepcopy(v, memo)
            setattr(result, k, new_val)
        return result

    def match_file(self, f):
        """
        Determine whether the passed file matches the Entity.

        Args:
            f (File): The BIDSFile instance to match against.

        Returns: the matched value if a match was found, otherwise None.
        """
        if self.regex is None:
            return None
        m = self.regex.search(f.path)
        val = m.group(1) if m is not None else None

        return self._astype(val)

    def unique(self):
        """ Returns all unique values/levels for the current entity. """
        return list(set(self.files.values()))

    def count(self, files=False):
        """ Returns a count of unique values or files.

        Args:
            files (bool): When True, counts all files mapped to the Entity.
                When False, counts all unique values.
        Returns: an int.
        """
        return len(self.files) if files else len(self.unique())

    def _astype(self, val):
        if val is not None and self.dtype is not None:
            val = self.dtype(val)
        return val


class Tag(Base):
    __tablename__ = 'tags'

    file_path = Column(String, ForeignKey('files.path'), primary_key=True)
    entity_name = Column(String, ForeignKey('entities.name'), primary_key=True)
    _value = Column(String, nullable=False)
    _dtype = Column(String, default='str')

    file = relationship('BIDSFile', backref=backref(
        "tags", collection_class=attribute_mapped_collection("entity_name")))
    entity = relationship('Entity', backref=backref(
        "tags", collection_class=attribute_mapped_collection("file_path")))

    def __init__(self, file, entity, value, dtype=None):

        if dtype is None:
            dtype = type(value)

        self.value = value

        if not isinstance(dtype, six.string_types):
            dtype = dtype.__name__
        if dtype not in ('str', 'float', 'int', 'bool'):
            # Try serializing to JSON first
            try:
                value = json.dumps(value)
                dtype = 'json'
            except:
                raise ValueError(
                    "Passed value has an invalid dtype ({}). Must be one of "
                    "int, float, bool, or 'str.".format(dtype))
        value = str(value)
        self.file_path = file.path
        self.entity_name = entity.name

        self._value = value
        self._dtype = dtype

        self._init_on_load()

    @reconstructor
    def _init_on_load(self):
        if self._dtype not in ('str', 'float', 'int', 'bool', 'json'):
            raise ValueError("Invalid dtype '{}'. Must be one of 'int', "
                             "'float', 'bool', 'str', or 'json'.".format(self._dtype))
        if self._dtype == 'json':
            self.value = json.loads(self._value)
            self.dtype = 'json'
        else:
            self.dtype = eval(self._dtype)
            self.value = self.dtype(self._value)


# Association objects
config_to_entity_map = Table('config_to_entity_map', Base.metadata,
    Column('config', String, ForeignKey('configs.name')),
    Column('entity', String, ForeignKey('entities.name'))
)
