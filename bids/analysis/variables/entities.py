from bids.utils import listify
from itertools import chain
from collections import namedtuple
from . import collections as clc


BASE_ENTITIES = ['subject', 'session', 'task', 'run']


class AnalysisLevel(object):

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
        ''' Return a flat list of all entities at the specified level.
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
        self.variables[variable.name] = variable

    def get_entities(self):
        entities = {} if self.parent is None else self.parent.get_entities()
        if self._level != 'dataset':
            entities[self._level] = self.id
        return entities


class Run(AnalysisLevel):

    _parent = 'session'
    _child = None

    def __init__(self, id, parent, image_file, duration,
                 repetition_time):
        self.image_file = image_file
        self.duration = duration
        self.repetition_time = repetition_time
        super(Run, self).__init__(id, parent)

    def get_info(self):
        return RunInfo(id=self.id, entities=self.get_entities(),
                       duration=self.duration, tr=self.repetition_time,
                       image=self.image_file)


RunInfo = namedtuple('RunInfo', ['id', 'entities', 'duration', 'tr', 'image'])


class Session(AnalysisLevel):

    _parent = 'subject'
    _child = 'run'


class Subject(AnalysisLevel):

    _parent = 'dataset'
    _child = 'session'


class Dataset(AnalysisLevel):

    _parent = None
    _child = 'subject'

    def __init__(self):
        super(Dataset, self).__init__(1, None)

    def get_variables(self, unit, variables=None, return_type='collection',
                      merge=False, sampling_rate=None, **selectors):
        ''' Retrieve variable data for a specified level in the Dataset.

        Args:
            unit (str): The unit of analysis to return variables for. Must be
                one of 'run', 'session', 'subject', or 'dataset'.
            variables (list): Optional list of variables names to return. If
                None, all available variables are returned.
            return_type (str): The type of returned object(s). Valid values:
                'collection': Returns BIDSVariableCollection(s)
                'df' or 'dataframe': Returns a pandas DataFrame
            merge (bool): If True, variables are merged across all observations
                of the current unit. E.g., if unit='subject' and return_type=
                'collection', variablesfrom all subjects will be merged into a
                single collection. If False, each observation is handled
                separately, and the result is returned as a list.
        '''

        return_type = return_type.lower()

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
            vs = clc.BIDSRunVariableCollection(vs, sampling_rate=sampling_rate) if unit == 'run' else clc.BIDSVariableCollection(vs)
            if return_type in ['df', 'dataframe']:
                vs = vs.to_df()
            results.append(vs)

        if merge:
            return results[0]

        return results

    def get_or_create_node(self, entities, *args, **kwargs):

        if 'run' in entities and 'session' not in entities:
            entities['session'] = 1

        return self._get_node(entities, *args, **kwargs)
