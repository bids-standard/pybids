from bids.utils import listify
from itertools import chain
from collections import namedtuple


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

    def _get_variables(self, level, ids=None, variables=None):

        if self._level == level:
            if variables is None:
                variables = list(self.variables.keys())
            return {k: v for k, v in self.variables.items() if k in variables}

        if level != self._child or ids is None:
            ids = self.children.keys()

        results = [self.children[c]._get_variables(level, ids, variables)
                   for c in ids]
        return chain(*results)

    def get_runs(self, **selectors):
        ''' Return a flat list of all Runs that match the selection criteria.
        '''
        if self._child is None:
            return [self]
        runs = []
        children = listify(selectors.get(self._child,
                                         list(self.children.keys())))
        for child_id in children:
            runs.extend(self.children[child_id].get_runs(**selectors))
        return runs

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

    def get_variables(self, level, ids=None, variables=None,
                      return_type='collection', merge=False):

        results = self._get_variables(level, ids, variables)

        # Convert dicts to Collections here
        collections = None

        if merge:
            pass

        if return_type == 'collection':
            return collections

        if return_type.lower() in ['df', 'dataframe']:
            dfs = [coll.to_df() for coll in listify(collections)]
            return dfs[0] if merge or level == 'dataset' else dfs

    def get_or_create_node(self, entities, *args, **kwargs):

        if 'run' in entities and 'session' not in entities:
            entities['session'] = 1

        return self._get_node(entities, *args, **kwargs)
