from itertools import chain
from collections import namedtuple
from . import kollekshuns as clc
import pandas as pd


class Node(object):
    ''' Base class for objects that represent a single object in the BIDS
    hierarchy.

    Args:
        id (int, str): A value uniquely identifying this node. Typically the
            entity value extracted from the filename via grabbids.
    '''

    def __init__(self, level, entities):
        self.level = level
        self.entities = entities
        self.variables = {}

    def add_variable(self, variable):
        ''' Adds a BIDSVariable to the current Node's list.

        Args:
            variable (BIDSVariable): The Variable to add to the list.
        '''
        self.variables[variable.name] = variable


class RunNode(Node):
    ''' Represents a single Run in a BIDS project.

    Args:
        id (int): The index of the run.
        entities (dict): Dictionary of entities for this Node.
        image_file (str): The full path to the corresponding nifti image.
        duration (float): Duration of the run, in seconds.
        repetition_time (float): TR for the run.
        task (str): The task name for this run.
    '''

    def __init__(self, entities, image_file, duration, repetition_time):
        self.image_file = image_file
        self.duration = duration
        self.repetition_time = repetition_time
        super(RunNode, self).__init__('run', entities)

    def get_info(self):

        return RunInfo(self.entities, self.duration, self.repetition_time,
                       self.image_file)


# Stores key information for each Run.
RunInfo = namedtuple('RunInfo', ['entities', 'duration', 'tr', 'image'])


class NodeIndex(Node):
    ''' Represents the top level in a BIDS hierarchy. '''

    def __init__(self):
        self.index = pd.DataFrame()
        self.nodes = []

    def get_collections(self, unit, names=None, merge=False,
                        sampling_rate=None, **entities):
        ''' Retrieve variable data for a specified level in the Dataset.

        Args:
            unit (str): The unit of analysis to return variables for. Must be
                one of 'run', 'session', 'subject', or 'dataset'.
            names (list): Optional list of variables names to return. If
                None, all available variables are returned.
            merge (bool): If True, variables are merged across all observations
                of the current unit. E.g., if unit='subject' and return_type=
                'collection', variablesfrom all subjects will be merged into a
                single collection. If False, each observation is handled
                separately, and the result is returned as a list.
            sampling_rate (int, str): If level='run', the sampling rate to
                pass onto the returned BIDSRunVariableCollection.
            entities: Optional constraints used to limit what gets returned.

        Returns:

        '''

        nodes = self.get_nodes(unit, entities)
        var_sets = []

        for n in nodes:
            var_set = list(n.variables.values())
            var_set = [v for v in var_set if v.matches_entities(entities)]
            if names is not None:
                var_set = [v for v in var_set if v.name in names]
            # Additional filtering on Variables past run level, because their
            # contents are extracted from TSV files containing rows from
            # multiple observations
            if unit != 'run':
                var_set = [v.filter(entities) for v in var_set]
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

    def get_nodes(self, level=None, entities=None, strict=False):

        entities = {} if entities is None else entities.copy()

        if level is not None:
            entities['level'] = level

        if entities is None:
            return self.nodes

        match_ents = set(entities.keys())
        common_cols = list(match_ents & set(self.index.columns))

        if strict and match_ents - common_cols:
            raise ValueError("Invalid entities: ", match_ents - common_cols)

        if not common_cols:
            return self.nodes

        # Construct query string that handles both single values and iterables
        query = []
        for col in common_cols:
            oper = 'in' if isinstance(entities[col], (list, tuple)) else '=='
            q = '{name} {oper} {val}'.format(name=col, oper=oper,
                                             val=repr(entities[col]))
            query.append(q)
        query = ' and '.join(query)

        rows = self.index.query(query)
        if rows.empty:
            return []

        # Sort and return
        sort_cols = ['subject', 'session', 'task', 'run']
        sort_cols = [sc for sc in sort_cols if sc in set(rows.columns)]
        sort_cols += list(set(rows.columns) - set(sort_cols))
        rows = rows.sort_values(sort_cols)
        inds = rows['node_index'].astype(int)
        return [self.nodes[i] for i in inds]

    def get_or_create_node(self, level, entities, *args, **kwargs):
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

        result = self.get_nodes(level, entities)

        if result:
            if len(result) > 1:
                raise ValueError("More than one matching Node found! If you're"
                                 " expecting more than one Node, use "
                                 "get_nodes() instead of get_or_create_node()."
                                 )
            return result[0]

        # Create Node
        if level == 'run':
            node = RunNode(entities, *args, **kwargs)
        else:
            node = Node(level, entities)

        entities = dict(entities, node_index=len(self.nodes), level=level)
        self.nodes.append(node)
        node_row = pd.Series(entities)
        self.index = self.index.append(node_row, ignore_index=True)

        return node
