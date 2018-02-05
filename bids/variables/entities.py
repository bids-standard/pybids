from bids.utils import listify
from itertools import chain
from collections import namedtuple
from . import collections as clc


BASE_ENTITIES = ['subject', 'session', 'task', 'run']


class Node(object):
    ''' Base class for objects that represent a single object in the BIDS
    hierarchy.

    Args:
        id (int, str): A value uniquely identifying this node. Typically the
            entity value extracted from the filename via grabbids.
        parent (Node): The parent Node.
        children (list): A list of child nodes.
    '''

    def __init__(self, id, parent=None, children=None, *args, **kwargs):
        self.id = id
        self.parent = parent
        self.children = children or {}
        self.variables = {}

    def __getitem__(self, key):
        return self.children[key]

    def _get_node(self, entities, *args, **kwargs):
        if self._child is None or self._child not in entities:
            return self

        child_id = entities[self._child]
        if child_id not in self.children:
            NodeClass = globals()[self._child.capitalize()]
            node = NodeClass(child_id, self, *args, **kwargs)
            self.children[child_id] = node
        node = self.children[child_id]
        return node._get_node(entities, *args, **kwargs)

    @property
    def _level(self):
        return self.__class__.__name__.lower()

    def get_nodes(self, level, **selectors):
        ''' Return a flat list of all Nodes at or below the current Node that
        match the specified criteria.

        Args:
            level (str): The target level of Node to return. Must be one of
                'dataset', 'subject', 'session', or 'run'.
            selectors: Optional keyword arguments placing constraints on what
                Nodes to return. Argument names be any of the standard
                entities in the hierarchy--i.e., 'subject', 'session', or
                'run'.

        Returns:
            A list of Nodes.
        '''
        if self._level == level:
            return [self]
        nodes = []
        children = listify(selectors.get(self._child,
                                         list(self.children.keys())))
        for child_id in children:
            nodes.extend(self.children[child_id].get_nodes(level, **selectors))
        return nodes

    def add_variable(self, variable):
        ''' Adds a BIDSVariable to the current Node's list.

        Args:
            variable (BIDSVariable): The Variable to add to the list.
        '''
        self.variables[variable.name] = variable

    def get_entities(self):
        ''' Returns a dictionary of entities for the current Node. '''
        entities = {} if self.parent is None else self.parent.get_entities()
        if self._level != 'dataset':
            entities[self._level] = self.id
        return entities


class Run(Node):
    ''' Represents a single Run in a BIDS project.

    Args:
        id (int): The index of the run.
        parent (Node): The parent Session.
        image_file (str): The full path to the corresponding nifti image.
        duration (float): Duration of the run, in seconds.
        repetition_time (float): TR for the run.
        task (str): The task name for this run.
    '''

    _parent = 'session'
    _child = None

    def __init__(self, id, parent, image_file, duration,
                 repetition_time, task):
        self.image_file = image_file
        self.duration = duration
        self.repetition_time = repetition_time
        self.task = task
        super(Run, self).__init__(id, parent)

    def get_info(self):
        entities = self.get_entities()
        entities['task'] = self.task
        return RunInfo(self.id, entities, self.duration, self.repetition_time,
                       self.image_file)


# Stores key information for each Run.
RunInfo = namedtuple('RunInfo', ['id', 'entities', 'duration', 'tr', 'image'])


class Session(Node):

    _parent = 'subject'
    _child = 'run'


class Subject(Node):

    _parent = 'dataset'
    _child = 'session'


class Dataset(Node):
    ''' Represents the top level in a BIDS hierarchy. '''
    _parent = None
    _child = 'subject'

    def __init__(self):
        super(Dataset, self).__init__(1, None)

    def get_collections(self, unit, variables=None, merge=False,
                        sampling_rate=None, **selectors):
        ''' Retrieve variable data for a specified level in the Dataset.

        Args:
            unit (str): The unit of analysis to return variables for. Must be
                one of 'run', 'session', 'subject', or 'dataset'.
            variables (list): Optional list of variables names to return. If
                None, all available variables are returned.
            merge (bool): If True, variables are merged across all observations
                of the current unit. E.g., if unit='subject' and return_type=
                'collection', variablesfrom all subjects will be merged into a
                single collection. If False, each observation is handled
                separately, and the result is returned as a list.
            sampling_rate (int, str): If level='run', the sampling rate to
                pass onto the returned BIDSRunVariableCollection.
            selectors: Optional constraints used to limit what gets returned.
                Valid argument names are 'run', 'session', and 'subject'.

        Returns:

        '''

        nodes = self.get_nodes(unit, **selectors)
        var_sets = []

        for n in nodes:
            var_set = list(n.variables.values())
            if variables is not None:
                var_set = [v for v in var_set if v.name in variables]
            var_sets.append(var_set)

        if merge:
            var_sets = [list(chain(*var_sets))]

        results = []
        for vs in var_sets:
            if not vs:
                continue
            if unit == 'run':
                vs = clc.BIDSRunVariableCollection(vs, sampling_rate)
            else:
                vs = clc.BIDSVariableCollection(vs)
            results.append(vs)

        if merge:
            return results[0]

        return results

    def get_or_create_node(self, entities, *args, **kwargs):
        ''' Retrieves a child Node based on the specified criteria, creating a
        new Node if necessary.

        Args:
            entities (dict): Dictionary of entities specifying which Node to
                return.
            args, kwargs: Optional positional or named arguments to pass onto
                class-specific initializers. These arguments are only used if
                a Node that matches the passed entities doesn't already exist,
                and a new one must be created.

        Returns:
            A Node instance.
        '''

        if 'run' in entities and 'session' not in entities:
            entities['session'] = 1

        return self._get_node(entities, *args, **kwargs)
