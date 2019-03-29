""" File-indexing functionality. """

import os
import re
import json
from keyword import iskeyword
import warnings
from copy import deepcopy

from .writing import build_path, write_contents_to_file
from .models import Config, BIDSFile, Entity, Tag
from ..utils import listify, check_path_matches_patterns
from ..config import get_option
from ..external import six


class BIDSNode(object):
    """ Represents a single directory or other logical grouping within a
    BIDS project.

    Args:
        path (str): The full path to the directory.
        config (str, list): One or more names of built-in configurations
            (e.g., 'bids' or 'derivatives') that specify the rules that apply
            to this node.
        root (BIDSNode): The node at the root of the tree the current node is
            part of.
        parent (BIDSNode): The parent of the current node.
        force_index (bool): Whether or not to forcibly index every file below
            this node, even if it fails standard BIDS validation.
    """

    _child_class = None
    _child_entity = None
    _entities = {}

    def __init__(self, path, config, root=None, parent=None,
                 force_index=False):
        self.path = path
        self.config = listify(config)
        self.root = root
        self.parent = parent
        self.entities = {}
        self.available_entities = {}
        self.children = []
        self.files = []
        self.variables = []
        self.force_index = force_index

        # Check for additional config file in directory
        layout_file = self.layout.config_filename
        config_file = os.path.join(self.abs_path, layout_file)
        if os.path.exists(config_file):
            cfg = Config.load(config_file)
            self.config.append(cfg)

        # Consolidate all entities
        self._update_entities()

        # Extract local entity values
        self._extract_entities()

        # Do subclass-specific setup
        self._setup()

        # Append to layout's master list of nodes
        self.layout.nodes.append(self)

        # Index files and create child nodes
        self.index()

    def __getitem__(self, key):
        if key in self.children:
            return self.children[key]
        if key in self.files:
            return self.files[key]
        raise AttributeError("BIDSNode at path {} has no child node or file "
                             "named {}.".format(self.path, key))
    
    def _update_entities(self):
        # Make all entities easily accessible in a single dict
        self.available_entities = {}
        for c in self.config:
            self.available_entities.update(c.entities)
    
    def _extract_entities(self):
        self.entities = {}
        for ent in self._entities:
            m = re.findall(self.available_entities[ent].pattern, self.path)
            if m:
                self.entities[ent] = m[0]

    def _get_child_class(self, path):
        """ Return the appropriate child class given a subdirectory path.
        
        Args:
            path (str): The path to the subdirectory.
        
        Returns: An uninstantiated BIDSNode or one of its subclasses.
        """
        if self._child_entity is None:
            return BIDSNode

        for i, child_ent in enumerate(listify(self._child_entity)):
            template = self.available_entities[child_ent].directory
            if template is None:
                return BIDSNode
            template = self.root_path + template
            # Construct regex search pattern from target directory template
            to_rep = re.findall(r'\{(.*?)\}', template)
            for ent in to_rep:
                patt = self.available_entities[ent].pattern
                template = template.replace('{%s}' % ent, patt)
            template += r'[^\%s]*$' % os.path.sep
            if re.match(template, path):
                return listify(self._child_class)[i]

        return BIDSNode

    def _setup(self):
        pass

    @property
    def abs_path(self):
        return os.path.join(self.root_path, self.path)

    @property
    def root_path(self):
        return self.path if self.root is None else self.root.path
    
    @property
    def layout(self):
        return self._layout if self.root is None else self.root.layout

    def index(self):
        """ Index all files/directories below the current BIDSNode. """

        config_list = self.config
        layout = self.layout

        from .layout import session

        for (dirpath, dirnames, filenames) in os.walk(self.path):

            # If layout configuration file exists, delete it
            layout_file = self.layout.config_filename
            if layout_file in filenames:
                filenames.remove(layout_file)

            for f in filenames:

                abs_fn = os.path.join(self.path, f)

                # Skip files that fail validation, unless forcibly indexing
                if not self.force_index and not layout._validate_file(abs_fn):
                    continue

                bf = BIDSFile(abs_fn)
                session.add(bf)

                # Extract entity values
                match_vals = {}
                for e in self.available_entities.values():
                    m = e.match_file(bf)
                    if m is None and e.mandatory:
                        break
                    if m is not None:
                        match_vals[e.name] = (e, m)

                # Create Entity <=> BIDSFile mappings
                if match_vals:
                    for name, (ent, val) in match_vals.items():
                        tag = Tag(bf, ent, str(val), ent._dtype)
                        session.add(tag)
                        # bf.entities[name] = val
                        # ent.add_file(bf.path, val)
                
                session.commit()

                self.files.append(bf)
                # Also add to the Layout's master list
                self.layout.files[bf.path] = bf

            root_node = self if self.root is None else self.root

            for d in dirnames:

                d = os.path.join(dirpath, d)

                # Derivative directories must always be added separately and
                # passed as their own root, so terminate if passed.
                if d.startswith(os.path.join(self.layout.root, 'derivatives')):
                    continue

                # Skip directories that fail validation, unless force_index
                # is defined, in which case we have to keep scanning, in the
                # event that a file somewhere below the current level matches.
                # Unfortunately we probably can't do much better than this
                # without a lot of additional work, because the elements of
                # .force_index can be SRE_Patterns that match files below in
                # unpredictable ways.
                if check_path_matches_patterns(d, self.layout.force_index):
                    self.force_index = True
                else:
                    valid_dir = layout._validate_dir(d)
                    # Note the difference between self.force_index and
                    # self.layout.force_index.
                    if not valid_dir and not self.layout.force_index:
                        continue

                child_class = self._get_child_class(d)
                # TODO: filter the config files based on include/exclude rules
                child = child_class(d, config_list, root_node, self,
                                    force_index=self.force_index)

                if self.force_index or valid_dir:
                    self.children.append(child)

            # prevent subdirectory traversal
            break


class BIDSSessionNode(BIDSNode):
    """ A BIDSNode associated with a single BIDS session. """

    _entities = {'subject', 'session'}

    def _setup(self):
        self.label = self.entities['session']


class BIDSSubjectNode(BIDSNode):
    """ A BIDSNode associated with a single BIDS subject. """

    _child_entity = 'session'
    _child_class = BIDSSessionNode
    _entities = {'subject'}

    def _setup(self):
        self.sessions = [c for c in self.children if
                         isinstance(c, BIDSSessionNode)]
        self.label = self.entities['subject']


class BIDSRootNode(BIDSNode):
    """ A BIDSNode representing the top level of an entire BIDS project. """

    _child_entity = 'subject'
    _child_class = BIDSSubjectNode

    def __init__(self, path, config, layout, force_index=False):
        self._layout = layout
        super(BIDSRootNode, self).__init__(path, config,
                                           force_index=force_index)
    
    def _setup(self):
        self.subjects = {c.label: c for c in self.children if
                         isinstance(c, BIDSSubjectNode)}
