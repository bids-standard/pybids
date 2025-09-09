""" Model classes used in BIDSLayouts. """

import re
import os
from pathlib import Path
from upath import UPath
import warnings
import json
from copy import deepcopy
from itertools import chain
from functools import lru_cache
from collections import UserDict

from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.orm.collections import attribute_mapped_collection
from sqlalchemy import Column, String, Boolean, ForeignKey, Table
from sqlalchemy.orm import reconstructor, relationship, backref, object_session

from bidsschematools import schema as bst_schema
from bidsschematools import rules
import re

try:
    from sqlalchemy.orm import declarative_base
except ImportError:  # sqlalchemy < 1.4
    from sqlalchemy.ext.declarative import declarative_base

from ..utils import listify
from .writing import build_path, write_to_file
from ..config import get_option
from .utils import BIDSMetadata, PaddedInt
from ..exceptions import BIDSChildDatasetError

Base = declarative_base()


class LayoutInfo(Base):
    """ Contains information about a BIDSLayout's initialization parameters."""

    __tablename__ = 'layout_info'

    root = Column(String, primary_key=True)
    absolute_paths = Column(Boolean, default=True)  # Removed, but may be in older DBs
    _derivatives = Column(String)
    _config = Column(String)

    def __init__(self, **kwargs):
        init_args = self._sanitize_init_args(kwargs)
        raw_cols = ['root']
        json_cols = ['derivatives', 'config']
        all_cols = raw_cols + json_cols
        missing_cols = set(all_cols) - set(init_args.keys())
        if missing_cols:
            raise ValueError("Missing mandatory initialization args: {}"
                             .format(missing_cols))
        for col in all_cols:
            setattr(self, col, init_args[col])
            if col in json_cols:
                json_data = json.dumps(init_args[col])
                setattr(self, '_' + col, json_data)

    @reconstructor
    def _init_on_load(self):
        if self.absolute_paths is False:
            warnings.warn(
                "PyBIDS database loaded with deprecated `absolute_paths` "
                "option. Returned paths will be absolute.",
                stacklevel=2,
            )
        for col in ['derivatives', 'config']:
            db_val = getattr(self, '_' + col)
            setattr(self, col, json.loads(db_val))

    def _sanitize_init_args(self, kwargs):
        """ Prepare initialization arguments for serialization """
        if 'root' in kwargs:
            kwargs['root'] = str(UPath(kwargs['root']).absolute())

        if 'config' in kwargs and isinstance(kwargs['config'], list):
            kwargs['config'] = [
                str(UPath(config).absolute())
                if isinstance(config, os.PathLike) else config
                for config in kwargs['config']
            ]

        # Get abspaths
        if kwargs.get('derivatives') not in (None, True, False):
            kwargs['derivatives'] = [
                str(UPath(der).absolute())
                for der in listify(kwargs['derivatives'])
                ]

        if kwargs.pop('absolute_paths', True) is not True:
            warnings.warn(
                "Deprecated `absolute_paths` option passed. "
                "Value must be True.", stacklevel=3,
            )

        return kwargs

    def __repr__(self):
        return f"<LayoutInfo {self.root}>"


class Config(Base):
    """Container for BIDS configuration information.

    Parameters
    ----------
    name : str
        The name to give the Config (e.g., 'bids').
    entities : list
        A list of dictionaries containing entity configuration
        information.
    default_path_patterns : list
        Optional list of patterns used to build new paths.
    session : :obj:`sqlalchemy.orm.session.Session` or None
        An optional SQLAlchemy session. If passed,
        the session is used to update the database with any newly created
        Entity objects. If None, no database update occurs.
    """
    __tablename__ = 'configs'

    name = Column(String, primary_key=True)
    _default_path_patterns = Column(String)
    entities = relationship(
        "Entity", secondary="config_to_entity_map",
        collection_class=attribute_mapped_collection('name'))

    def __init__(self, name, entities=None, default_path_patterns=None,
                 session=None):

        self.name = name
        self.default_path_patterns = default_path_patterns
        self._default_path_patterns = json.dumps(default_path_patterns)

        if entities:
            for ent in entities:
                if session is not None:
                    existing = (session.query(Config)
                                .filter_by(name=ent['name']).first())
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
    def load(cls, config, session=None):
        """Load a Config instance from the passed configuration data.

        Parameters
        ----------
        config : str or dict
            A string or dict containing configuration information.
            Must be one of:
            * A string giving the name of a predefined config file
                (e.g., 'bids' or 'derivatives')
            * A path to a JSON file containing config information
            * A dictionary containing config information
            * 'bids-schema' to load from the official BIDS schema
            * dict with 'schema_version' key to load specific schema version
        session : :obj:`sqlalchemy.orm.session.Session` or None
            An optional SQLAlchemy Session instance.
            If passed, the session is used to check the database for (and
            return) an existing Config with name defined in config['name'].

        Returns
        -------
        A Config instance.
        """

        # Handle schema-based config loading
        if config == 'bids-schema':
            return cls._from_schema(session=session)
        elif isinstance(config, dict) and 'schema_version' in config:
            return cls._from_schema(
                schema_version=config['schema_version'],
                session=session
            )

        # Existing JSON/file-based loading
        if isinstance(config, (str, Path, UPath)):
            config_paths = get_option('config_paths')
            if config in config_paths:
                config = config_paths[config]
            if not UPath(config).exists():
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

    @classmethod
    def _from_schema(cls, schema_version=None, session=None):
        """Load config from BIDS schema - store patterns directly."""
        from bidsschematools import rules, schema as bidsschema
        
        bids_schema = bidsschema.load_schema()
        
        # Collect ALL patterns from bidsschematools (keep existing collection logic)
        all_patterns = []
        file_sections = {
            'raw': bids_schema.rules.files.raw,
            'deriv': bids_schema.rules.files.deriv,  
            'common': bids_schema.rules.files.common
        }
        
        for section_type, sections in file_sections.items():
            if section_type == 'raw':
                for datatype_name in sections.keys():
                    datatype_rules = getattr(sections, datatype_name)
                    regex_rules = rules.regexify_filename_rules(datatype_rules, bids_schema, level=1)
                    all_patterns.extend(regex_rules)
            else:
                for subsection_name in sections.keys():
                    subsection_rules = getattr(sections, subsection_name)
                    regex_rules = rules.regexify_filename_rules(subsection_rules, bids_schema, level=1)
                    all_patterns.extend(regex_rules)
        
        # Store patterns directly - NO PARSING EVER
        config_name = f'bids-schema-{bids_schema.bids_version}'
        
        # Extract entities directly from schema rules - no regex parsing!
        entity_names = cls._extract_entity_names_from_rules(bids_schema)
        
        # Extract actual values for key entities from schema rules directly
        entity_values = cls._extract_entity_values_from_rules(bids_schema, entity_names)
        
        # Create Entity objects
        entities = cls._create_entities_from_schema(entity_names, entity_values, bids_schema)
        
        config = cls(name=config_name, entities=entities, session=session)
        
        return config
    
    @classmethod
    def _extract_entity_names_from_rules(cls, bids_schema):
        """Extract entity names directly from schema rules - no regex parsing!"""
        
        # Get entity names from all rule sections
        entity_names = set()
        file_sections = {
            'raw': bids_schema.rules.files.raw,
            'deriv': bids_schema.rules.files.deriv,  
            'common': bids_schema.rules.files.common
        }
        
        for section_type, sections in file_sections.items():
            for section_name in sections.keys():
                section_rules = getattr(sections, section_name)
                for rule_name in section_rules.keys():
                    rule = getattr(section_rules, rule_name)
                    rule_dict = dict(rule)
                    if 'entities' in rule_dict:
                        entity_names.update(rule_dict['entities'].keys())
        
        # Add special constructs that appear in patterns but aren't "entities"
        entity_names.update(['suffix', 'extension', 'datatype'])
        
        return entity_names
    
    @classmethod
    def _create_entities_from_schema(cls, entity_names, entity_values, bids_schema):
        """Create Entity objects from schema information."""
        entities = []
        
        for entity_name in sorted(entity_names):
            # Get entity info from schema if available
            entity_spec = bids_schema.objects.entities.get(entity_name, {})
            
            # Handle special entities that don't have BIDS prefixes
            special_entities = {'suffix', 'extension', 'datatype'}
            
            if entity_name in special_entities:
                # These entities don't have prefixes - they appear directly
                if entity_name in entity_values and entity_values[entity_name]:
                    # Filter out problematic values for extension entity
                    if entity_name == 'extension':
                        # Remove empty string and wildcard patterns that would match everything
                        filtered_values = {v for v in entity_values[entity_name] 
                                         if v and v not in ('.*', '**', '*')}
                        values = '|'.join(sorted(filtered_values))
                    else:
                        values = '|'.join(sorted(entity_values[entity_name]))
                    
                    if entity_name == 'extension':
                        pattern = fr'({values})\Z'  # Extensions are at the end
                    elif entity_name == 'datatype':
                        pattern = fr'/({values})/'  # Datatypes are directory names  
                    else:  # suffix
                        pattern = fr'[_/\\\\]({values})\.?'  # Before extension, with optional dot
            else:
                # Regular entities - use schema format patterns directly!
                bids_prefix = entity_spec.get('name', entity_name)
                format_type = entity_spec.get('format', 'label')
                
                # Get the official value pattern from schema formats
                if format_type in bids_schema.objects.formats:
                    format_obj = dict(bids_schema.objects.formats[format_type])
                    value_pattern = format_obj['pattern']  # [0-9a-zA-Z]+, [0-9]+, etc.
                
                # Combine prefix + schema format pattern
                if entity_name in ['subject', 'session']:
                    pattern = fr'[/\\\\]+{bids_prefix}-({value_pattern})'
                else:
                    pattern = fr'[_/\\\\]+{bids_prefix}-({value_pattern})'
            
            entity_config = {
                'name': entity_name,
                'pattern': pattern,
                'dtype': 'int' if entity_spec.get('format') == 'index' else 'str'
            }
            
            entities.append(entity_config)
        
        return entities
    
    @classmethod
    def _extract_entity_values_from_rules(cls, bids_schema, entity_names):
        """Extract entity values directly from schema rules - no regex parsing!"""
        
        entity_values = {name: set() for name in entity_names}
        file_sections = {
            'raw': bids_schema.rules.files.raw,
            'deriv': bids_schema.rules.files.deriv,  
            'common': bids_schema.rules.files.common
        }
        
        for section_type, sections in file_sections.items():
            for section_name in sections.keys():
                section_rules = getattr(sections, section_name)
                for rule_name in section_rules.keys():
                    rule = getattr(section_rules, rule_name)
                    rule_dict = dict(rule)
                    
                    # Get values for special entities directly from rules
                    if 'suffixes' in rule_dict:
                        entity_values['suffix'].update(rule_dict['suffixes'])
                    if 'extensions' in rule_dict:
                        entity_values['extension'].update(rule_dict['extensions'])
                    if 'datatypes' in rule_dict:
                        entity_values['datatype'].update(rule_dict['datatypes'])
                    
                    # For regular entities, we don't extract specific values
                    # They use generic patterns like [0-9a-zA-Z]+ 
        
        return entity_values
    
    
    
    @staticmethod 
    def _extract_separator_from_context(entity_name, before_context, after_context, full_pattern, schema):
        """Extract separator pattern from schema context.
        
        Analyzes the actual schema pattern context to determine what
        separators are used for this entity by examining the schema regex
        and entity classification.
        """
        # Find the entity in the full pattern and extract its actual context
        entity_match = re.search(rf'(\S*?)(\(\?P<{entity_name}>[^)]+\))', full_pattern)
        if not entity_match:
            return f"[_/\\\\]+{entity_name}-"  # Fallback
        
        prefix_context = entity_match.group(1)
        
        # Determine entity type from schema - is it a real entity or special construct?
        is_real_entity = entity_name in schema.objects.entities
        
        if not is_real_entity:
            # This is a schema construct (extension/suffix/datatype capturing group)
            # Analyze the pattern context to determine positioning rules
            
            # Analyze from the actual pattern context
            if '/' in before_context and '/' in after_context:
                # Appears between directory separators (like datatype)
                return "[/\\\\]+"
            elif before_context.endswith(')'):
                # Comes directly after another capturing group (like extension after suffix)
                return ""
            elif '_)?' in before_context or 'chunk)_)?' in before_context:
                # Comes after underscore separators from entities (like suffix)
                return "[_/\\\\]+"
            else:
                # Analyze position in full pattern to determine context
                entity_pos = full_pattern.find(f'(?P<{entity_name}>')
                pattern_before = full_pattern[:entity_pos]
                
                # Count directory structure elements before this construct
                slash_count = pattern_before.count('/')
                
                if slash_count <= 2:
                    # In directory part - use slash separators
                    return "[/\\\\]+"
                else:
                    # In filename part - analyze further
                    if ')(?P<' in pattern_before[-20:]:
                        # Directly after another construct
                        return ""
                    else:
                        # After separator
                        return "[_/\\\\]+"
        
        # Extract the actual prefix used in the schema pattern  
        # Look for patterns like 'sub-(?P<subject>' or '_task-(?P<task>'
        prefix_pattern = re.search(rf'([/_]?)([a-zA-Z]+)-\(\?P<{entity_name}>', full_pattern)
        if prefix_pattern:
            separator_char = prefix_pattern.group(1)
            prefix = prefix_pattern.group(2)
            
            if separator_char == '/':
                return f"[/\\\\]+{prefix}-"
            elif separator_char == '_':
                return f"[_/\\\\]+{prefix}-" 
            else:  # Direct prefix (no separator before)
                # Check position to determine separator type
                entity_pos = full_pattern.find(f'(?P<{entity_name}>')
                pattern_before = full_pattern[:entity_pos]
                slash_count = pattern_before.count('/')
                
                if slash_count <= 2:  # Directory part
                    return f"[/\\\\]+{prefix}-"
                else:  # Filename part  
                    return f"[_/\\\\]+{prefix}-"
        
        # Try to extract any entity prefix from before_context
        prefix_match = re.search(r'([a-zA-Z]+)-$', before_context)
        if prefix_match:
            prefix = prefix_match.group(1)
            # Verify this is a real entity prefix from schema
            for entity in schema.objects.entities.values():
                if entity.get('name') == prefix:
                    if '/' in before_context:
                        return f"[/\\\\]+{prefix}-"
                    else:
                        return f"[_/\\\\]+{prefix}-"
        
        # For entities with prefixes, extract from schema pattern
        broad_match = re.search(rf'([a-zA-Z]+)-\(\?P<{entity_name}>', full_pattern)
        if broad_match:
            prefix = broad_match.group(1)
            
            # Determine separator type from position in pattern
            entity_pos = full_pattern.find(f'(?P<{entity_name}>')
            pattern_before_entity = full_pattern[:entity_pos]
            slash_count = pattern_before_entity.count('/')
            
            if slash_count <= 2:  # Directory part
                return f"[/\\\\]+{prefix}-"
            else:  # Filename part
                return f"[_/\\\\]+{prefix}-"
        
        # If no prefix pattern found, this entity shouldn't have a prefix
        # This handles cases where entities appear directly without prefix-
        return f"[_/\\\\]+{entity_name}-"  # Final fallback
    
    @staticmethod
    def _extract_directory_config(entity_name, pattern, schema):
        """Extract directory configuration from schema directory rules."""
        
        # Get directory rules from schema
        if not hasattr(schema.rules, 'directories') or not hasattr(schema.rules.directories, 'raw'):
            return None
            
        dir_rules = schema.rules.directories.raw
        
        # Check if this entity has a directory rule
        for rule_name, rule in dir_rules.items():
            rule_dict = dict(rule)
            if rule_dict.get('entity') == entity_name:
                # This entity creates directories
                
                # Build directory template based on schema hierarchy
                # Find parent entities by checking what directories can contain this entity
                parent_entities = []
                
                # Check all directory rules to see which ones list this entity in subdirs
                for parent_rule_name, parent_rule in dir_rules.items():
                    parent_rule_dict = dict(parent_rule)
                    subdirs = parent_rule_dict.get('subdirs', [])
                    
                    # Check if this entity is in parent's subdirs
                    for subdir in subdirs:
                        if isinstance(subdir, dict) and 'oneOf' in subdir:
                            if entity_name in subdir['oneOf']:
                                if parent_rule_dict.get('entity'):
                                    parent_entities.append(parent_rule_dict['entity'])
                        elif subdir == entity_name:
                            if parent_rule_dict.get('entity'):
                                parent_entities.append(parent_rule_dict['entity'])
                
                # Build template with parent hierarchy
                template_parts = []
                template_parts.extend([f'{{{parent}}}' for parent in parent_entities])
                template_parts.append(f'{{{entity_name}}}')
                
                return ''.join(template_parts)
        
        return None

    @staticmethod
    def _entity_appears_in_all_modalities(entity_name, schema):
        """Check if an entity appears in ALL modalities in the schema.
        
        Entities that appear in all modalities (like suffix, extension) should have
        their values combined across all modalities to ensure complete coverage.
        
        Parameters
        ----------
        entity_name : str
            The entity name to check
        schema : object
            The BIDS schema object
            
        Returns
        -------
        bool
            True if entity appears in all modalities, False otherwise
        """
        # Get all modalities
        all_modalities = list(schema.rules.files.raw.keys())
        if not all_modalities:
            return False
            
        modalities_with_entity = set()
        
        # Check each modality for this entity
        for modality in all_modalities:
            datatype_rules = getattr(schema.rules.files.raw, modality)
            patterns = rules.regexify_filename_rules(datatype_rules, schema, level=1)
            
            # Check if entity appears in any pattern for this modality
            for pattern in patterns:
                entities_in_pattern = re.findall(r'\(\?P<([^>]+)>', pattern['regex'])
                if entity_name in entities_in_pattern:
                    modalities_with_entity.add(modality)
                    break  # Found in this modality, move to next
        
        # Return True if entity appears in ALL modalities
        return len(modalities_with_entity) == len(all_modalities)



    def __repr__(self):
        return f"<Config {self.name}>"


class BIDSFile(Base):
    """Represents a single file or directory in a BIDS dataset.

    Parameters
    ----------
    filename : str
        The path to the corresponding file.
    """
    __tablename__ = 'files'

    path = Column(String, primary_key=True)
    filename = Column(String)
    dirname = Column(String)
    entities = association_proxy("tags", "value")
    is_dir = Column(Boolean, index=True)
    class_ = Column(String(20))

    _associations = relationship('BIDSFile', secondary='associations',
                                 primaryjoin='FileAssociation.dst == BIDSFile.path',
                                 secondaryjoin='FileAssociation.src == BIDSFile.path')

    __mapper_args__ = {
        'polymorphic_on': class_,
        'polymorphic_identity': 'file'
    }

    def __init__(self, filename):
        self.path = str(filename)
        self.filename = self._path.name
        self.dirname = str(self._path.parent)
        self.is_dir = not self.filename

    @property
    def _path(self):
        return UPath(self.path)

    @property
    def _dirname(self):
        return UPath(self.dirname)

    def __repr__(self):
        return "<{} filename='{}'>".format(self.__class__.__name__, self.path)

    def __fspath__(self):
        return self.path

    @property
    @lru_cache()
    def relpath(self):
        """Return path relative to layout root"""
        root = object_session(self).query(LayoutInfo).first().root
        return str(UPath(self.path).relative_to(root))

    def get_associations(self, kind=None, include_parents=False):
        """Get associated files, optionally limiting by association kind.

        Parameters
        ----------
        kind : str
            The kind of association to return (e.g., "Child").
            By default, all associations are returned.
        include_parents : bool
            If True, files related through inheritance
            are included in the returned list. If False, only directly
            associated files are returned. For example, a file's JSON
            sidecar will always be returned, but other JSON files from
            which the sidecar inherits will only be returned if
            include_parents=True.

        Returns
        -------
        list
            A list of BIDSFile instances.
        """
        if kind is None and not include_parents:
            return self._associations

        session = object_session(self)
        q = (session.query(BIDSFile)
             .join(FileAssociation, BIDSFile.path == FileAssociation.dst)
             .filter_by(src=self.path))

        if kind is not None:
            q = q.filter_by(kind=kind)

        associations = q.all()

        if not include_parents:
            return associations

        def collect_associations(results, bidsfile):
            results.append(bidsfile)
            for p in bidsfile.get_associations('Child'):
                results = collect_associations(results, p)
            return results

        return list(chain(*[collect_associations([], bf) for bf in associations]))

    def get_metadata(self):
        """Return all metadata associated with the current file. """
        md = BIDSMetadata(self.path)
        md.update(self.get_entities(metadata=True))
        return md

    def get_entities(self, metadata=False, values='tags'):
        """Return entity information for the current file.

        Parameters
        ----------
        metadata : bool or None
            If False (default), only entities defined
            for filenames (and not those found in the JSON sidecar) are
            returned. If True, only entities found in metadata files (and not
            defined for filenames) are returned. If None, all available
            entities are returned.
        values : str
            The kind of object to return in the dict's values.
            Must be one of:
                * 'tags': Returns only the tagged value--e.g., if the key
                is "subject", the value might be "01".
                * 'objects': Returns the corresponding Entity instance.

        Returns
        -------
        dict
            A dict, where keys are entity names and values are Entity
            instances.
        """
        if metadata is None and values == 'tags':
            return self.entities

        session = object_session(self)
        query = (session.query(Tag)
                 .filter_by(file_path=self.path)
                 .join(Entity))

        if metadata not in (None, 'all'):
            query = query.filter(Tag.is_metadata == metadata)

        results = query.all()
        if values.startswith('obj'):
            return {t.entity_name: t.entity for t in results}
        return {t.entity_name: t.value for t in results}

    def copy(self, path_patterns, symbolic_link=False, root=None,
             conflicts='fail'):
        """Copy the contents of a file to a new location.

        Parameters
        ----------
        path_patterns : list
            List of patterns used to construct the new
            filename. See :obj:`build_path` documentation for details.
        symbolic_link : bool
            If True, use a symbolic link to point to the
            existing file. If False, creates a new file.
        root : str
            Optional path to prepend to the constructed filename.
        conflicts : str
            Defines the desired action when the output path already exists.
            Must be one of:
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

        if self._path.is_absolute() or root is None:
            path = self._path
        else:
            path = UPath(root) / self._path

        if not path.exists():
            raise ValueError("Target filename to copy/symlink (%s) doesn't "
                             "exist." % path)

        kwargs = dict(path=new_filename, root=root, conflicts=conflicts)
        if symbolic_link:
            kwargs['link_to'] = path
        else:
            kwargs['copy_from'] = path

        write_to_file(**kwargs)


class BIDSDataFile(BIDSFile):
    """Represents a single data file in a BIDS dataset.

    Derived from `BIDSFile` and provides additional functionality such as
    obtaining pandas DataFrame data representation (via `get_df`).
    """

    __mapper_args__ = {
        'polymorphic_identity': 'data_file'
    }

    def get_df(self, include_timing=True, adjust_onset=False,
               enforce_dtypes=True, **pd_args):
        """Return the contents of a tsv file as a pandas DataFrame.

        Parameters
        ----------
        include_timing : bool
            If True, adds an "onset" column to dense
            timeseries files (e.g., *_physio.tsv.gz).
        adjust_onset : bool
            If True, the onset of each sample in a dense
            timeseries file is shifted to reflect the "StartTime" value in
            the JSON sidecar. If False, the first sample starts at 0 secs.
            Ignored if include_timing=False.
        enforce_dtypes : bool
            If True, enforces the data types defined in
            the BIDS spec on core columns (e.g., subject_id and session_id
            must be represented as strings).
        pd_args : dict
            Optional keyword arguments to pass onto pd.read_csv().

        Returns
        -------
        :obj:`pandas.DataFrame`
            A pandas DataFrame.
        """
        import pandas as pd
        import numpy as np

        if enforce_dtypes:
            dtype = {
                'subject_id': str,
                'session_id': str,
                'participant_id': str
            }
        else:
            dtype = None

        # TODO: memoize this for efficiency. (Note: caching is insufficient,
        # because the dtype enforcement will break if we ignore the value of
        # enforce_dtypes).
        suffix = self.entities['suffix']
        header = None if suffix in {'physio', 'stim'} else 'infer'
        self.data = pd.read_csv(self.path, sep='\t', na_values='n/a',
                                dtype=dtype, header=header, **pd_args)

        data = self.data.copy()

        if self.entities['extension'] == '.tsv.gz':
            md = self.get_metadata()
            # We could potentially include some validation here, but that seems
            # like a job for the BIDS Validator.
            data.columns = md['Columns']
            if include_timing:
                onsets = np.arange(len(data)) / md['SamplingFrequency']
                if adjust_onset:
                    onsets += md['StartTime']
                data.insert(0, 'onset', onsets)

        return data


class BIDSImageFile(BIDSFile):
    """Represents a single neuroimaging data file in a BIDS dataset.

    Derived from `BIDSFile` and provides additional functionality such as
    obtaining nibabel's image file representation (via `get_image`).
    """

    __mapper_args__ = {
        'polymorphic_identity': 'image_file'
    }

    def get_image(self, **kwargs):
        """Return the associated image file (if it exists) as a NiBabel object

        Any keyword arguments are passed to ``nibabel.load``.
        """
        try:
            import nibabel as nb
            return nb.load(self.path, **kwargs)
        except Exception as e:
            raise ValueError("'{}' does not appear to be an image format "
                             "NiBabel can read.".format(self.path)) from e


class BIDSJSONFile(BIDSFile):
    """Represents a single JSON metadata file in a BIDS dataset.

    Derived from `BIDSFile` and provides additional functionality for reading
    the contents of JSON files as either dicts or strings.
    """
    __mapper_args__ = {
        'polymorphic_identity': 'json_file'
    }

    def get_dict(self):
        """Return the contents of the current file as a dictionary. """
        d = json.loads(self.get_json())
        if not isinstance(d, dict):
            raise ValueError("File %s is a json containing %s, not a dict which was expected" % (self.path, type(d)))
        return d

    def get_json(self):
        """Return the contents of the current file as a JSON string. """
        with open(self.path, 'r') as f:
            return f.read()


class Entity(Base):
    """
    Represents a single entity defined in the JSON config.

    Parameters
    ----------
    name : str
        The name of the entity (e.g., 'subject', 'run', etc.)
    pattern : str
        A regex pattern used to match against file names.
        Must define at least one group, and only the first group is
        kept as the match.
    mandatory : bool
        If True, every File _must_ match this entity.
    directory : str
        Optional pattern defining a directory associated
        with the entity.
    dtype : str
        The optional data type of the Entity values. Must be
        one of 'int', 'float', 'bool', or 'str'. If None, no type
        enforcement will be attempted, which means the dtype of the
        value may be unpredictable.
    """
    __tablename__ = 'entities'

    name = Column(String, primary_key=True)
    mandatory = Column(Boolean, default=False)
    pattern = Column(String)
    directory = Column(String, nullable=True)
    _dtype = Column(String, default='str')
    files = association_proxy("tags", "value")

    def __init__(self, name, pattern=None, mandatory=False, directory=None,
                 dtype='str'):
        self.name = name
        self.pattern = pattern
        self.mandatory = mandatory
        self.directory = directory

        if not isinstance(dtype, str):
            dtype = dtype.__name__
        self._dtype = dtype

        self._init_on_load()

    def __repr__(self):
        return f"<Entity {self.name} (pattern={self.pattern}, dtype={self.dtype})>"

    @reconstructor
    def _init_on_load(self):
        if self._dtype not in ('str', 'float', 'int', 'bool'):
            raise ValueError("Invalid dtype '{}'. Must be one of 'int', "
                             "'float', 'bool', or 'str'.".format(self._dtype))
        if self._dtype == "int":
            self.dtype = PaddedInt
        else:
            self.dtype = eval(self._dtype)
        self.regex = re.compile(self.pattern) if self.pattern is not None else None

    def __iter__(self):
        yield from self.unique()

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)

        # Directly copy the SQLAlchemy connection before any setattr calls,
        # otherwise failures occur sporadically on Python 3.5 when the
        # _sa_instance_state attribute (randomly!) disappears.
        result._sa_instance_state = self._sa_instance_state

        memo[id(self)] = result

        for k, v in self.__dict__.items():
            if k == '_sa_instance_state':
                continue
            new_val = getattr(self, k) if k == 'regex' else deepcopy(v, memo)
            setattr(result, k, new_val)
        return result

    def match_file(self, f):
        """
        Determine whether the passed file matches the Entity.

        Parameters
        ----------
        f : BIDSFile
            The BIDSFile instance to match against.

        Returns
        -------
        the matched value if a match was found, otherwise None.
        """
        if self.regex is None:
            return None
        m = self.regex.search(f.path)
        val = m.group(1) if m is not None else None

        return self._astype(val)

    def unique(self):
        """Return all unique values/levels for the current entity.
        """
        return list(set(self.files.values()))

    def count(self, files=False):
        """Return a count of unique values or files.

        Parameters
        ----------
        files : bool
            When True, counts all files mapped to the Entity.
            When False, counts all unique values.

        Returns
        -------
        int
            Count of unique values or files.
        """
        return len(self.files) if files else len(self.unique())

    def _astype(self, val):
        if val is not None and self.dtype is not None:
            val = self.dtype(val)
        return val

type_map = {
    'str': str,
    'int': PaddedInt,
    'float': float,
    'bool': bool,
    'json': 'json',
}
class Tag(Base):
    """Represents an association between a File and an Entity.

    Parameters
    ----------
    file : BIDSFile
        The associated BIDSFile.
    entity : Entity
        The associated Entity.
    value : json-serializable type
        The value to store for this file/entity pair. Must be of type
        str, int, float, bool, or any json-serializable structure.
    dtype : str
        Optional type for the value field. If None, inferred from
        value. If passed, must be one of str, int, float, bool, or json.
        Any other value will be treated as json (and will fail if the
        value can't be serialized to json).
    is_metadata : bool
        Indicates whether or not the Entity is derived
        from JSON sidecars (True) or is a predefined Entity from a
        config (False).
    """
    __tablename__ = 'tags'

    file_path = Column(String, ForeignKey('files.path'), primary_key=True)
    entity_name = Column(String, ForeignKey('entities.name'), primary_key=True)
    _value = Column(String, nullable=False)
    _dtype = Column(String, default='str')
    is_metadata = Column(Boolean, default=False)


    file = relationship('BIDSFile', backref=backref(
        "tags", collection_class=attribute_mapped_collection("entity_name")))
    entity = relationship('Entity', backref=backref(
        "tags", collection_class=attribute_mapped_collection("file_path")))

    def __init__(self, file, entity, value, dtype=None, is_metadata=False):
        data = _create_tag_dict(file, entity, value, dtype, is_metadata)

        self.file_path = data['file_path']
        self.entity_name = data['entity_name']
        self._dtype = data['_dtype']
        self._value = data['_value']
        self.is_metadata = data['is_metadata']

        self.dtype = type_map[self._dtype]
        if self._dtype != 'json':
            self.value = self.dtype(value)
        else:
            self.value = value

    def __repr__(self):
        msg = "<Tag file:{!r} entity:{!r} value:{!r}>"
        return msg.format(self.file_path, self.entity_name, self.value)

    @reconstructor
    def _init_on_load(self):
        if self._dtype == 'json':
            self.value = json.loads(self._value)
            self.dtype = 'json'
        elif self._dtype == 'bool':
            self.value = self._value == 'True'
            self.dtype = bool
        else:
            self.dtype = type_map[self._dtype]
            self.value = self.dtype(self._value)

def _create_tag_dict(file, entity, value, dtype=None, is_metadata=False):
        data = {}
        if dtype is None:
            dtype = type(value)

        if not isinstance(dtype, str):
            dtype = dtype.__name__

        if dtype in ['list', 'dict']:
            _dtype = 'json'
            _value = json.dumps(value)
        else:
            _dtype = dtype
            _value = str(value)
        if _dtype not in ('str', 'float', 'int', 'bool', 'json'):
            raise ValueError(
                f"Passed value has an invalid dtype ({dtype}). Must be one of "
                "int, float, bool, or str.")

        data['is_metadata'] = is_metadata
        data['file_path'] = file.path
        data['entity_name'] = entity.name
        data['_dtype'] = _dtype
        data['_value'] = _value

        return data


class FileAssociation(Base):
    __tablename__ = 'associations'

    src = Column(String, ForeignKey('files.path'), primary_key=True)
    dst = Column(String, ForeignKey('files.path'), primary_key=True)
    kind = Column(String, primary_key=True)


# Association objects
config_to_entity_map = Table('config_to_entity_map', Base.metadata,
                             Column('config', String, ForeignKey('configs.name')),
                             Column('entity', String, ForeignKey('entities.name'))
                             )


class DerivativeDatasets(UserDict):
    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            pass

        try:
            result = self.get_pipeline(key)
            warnings.warn(
                "Directly selecting derivative datasets using "
                "pipeline name (i.e. dataset.derivatives[<pipeline_name>] will be "
                "phased out in an upcoming release. Select instead using the folder "
                "name of the dataset (i.e. dataset.derivatives[<folder_name>]), or use "
                "dataset.derivatives.get_pipeline(<pipeline_name>).",
                DeprecationWarning,
            )
            return result
        except KeyError as err:
            raise KeyError(
                f"No datasets found matching {key} either as a pipeline name or as "
                "a dataset file name."
            ) from err


    def get_pipeline(self, pipeline):
        matches = {
            (name, dataset) for name, dataset in self.data.items()
            if dataset.source_pipeline == pipeline
        }
        if len(matches) > 1:
            datasets = "\n\t- ".join(match[0] for match in matches)
            raise BIDSChildDatasetError(
                f"Multiple datasets generated by {pipeline} were found:\n"
                f"\t- {datasets}\n\n"
                "Select a specific dataset by using "
                "dataset.derivatives[<dataset_folder_name>]."
            )
        if not matches:
            raise KeyError(f"No match found for {pipeline}")
        return next(iter(matches))[1]
