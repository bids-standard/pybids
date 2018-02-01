import numpy as np
import pandas as pd
import math
from copy import deepcopy
from abc import abstractproperty, abstractmethod, ABCMeta
from scipy.interpolate import interp1d
from bids.utils import listify
from itertools import chain

# from .collections import _build_dense_index


BASE_ENTITIES = ['subject', 'session', 'task', 'run']


class AnalysisLevel(object):

    def __init__(self, id, parent=None, children=None, *args, **kwargs):
        self.id = id
        self.parent = parent
        self.children = children or {}
        self.variables = {}

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


class Run(AnalysisLevel):

    _parent = 'session'
    _child = None

    def __init__(self, id, parent, image_file, duration,
                 repetition_time):
        self.image_file = image_file
        self.duration = duration
        self.repetition_time = repetition_time
        self.dense_index = None
        super(Run, self).__init__(id, parent)


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

        if return_type == 'dict':
            return results

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


class BIDSColumn(object):

    ''' Base representation of a column in a BIDS project. '''

    __metaclass__ = ABCMeta

    def __init__(self, name, values):
        self.name = name
        self.values = values

    def clone(self, data=None, **kwargs):
        ''' Clone (deep copy) the current column, optionally replacing its
        data and/or any other attributes.
        Args:
            data (DataFrame, ndarray): Optional new data to substitute into
                the cloned column. Must have same dimensionality as the
                original.
            kwargs (dict): Optional keyword arguments containing new attribute
                values to set in the copy. E.g., passing `name='my_name'`
                would set the `.name` attribute on the cloned instance to the
                passed value.
        '''
        result = deepcopy(self)
        if data is not None:
            if data.shape != self.values.shape:
                raise ValueError("Replacement data has shape %s; must have "
                                 "same shape as existing data %s." %
                                 (data.shape, self.values.shape))
            result.values = pd.Series(data)

        if kwargs:
            for k, v in kwargs.items():
                setattr(result, k, v)

        # Need to update name on Series as well
        result.values.name = kwargs.get('name', self.name)
        return result

    @abstractmethod
    def aggregate(self, unit, level, func):
        pass

    @abstractproperty
    def index(self):
        pass

    def get_grouper(self, groupby='unique_run_id'):
        ''' Return a pandas Grouper object suitable for use in groupby calls.
        Args:
            groupby (str, list): Name(s) of column(s) defining the grouper
                object. Anything that would be valid inside a .groupby() call
                on a pandas structure.
        Returns:
            A pandas Grouper object constructed from the specified columns
                of the current index.
        '''
        return pd.core.groupby._get_grouper(self.index, groupby)[0]

    def apply(self, func, groupby='unique_run_id', *args, **kwargs):
        ''' Applies the passed function to the groups defined by the groupby
        argument. Works identically to the standard pandas df.groupby() call.
        Args:
            func (callable): The function to apply to each group.
            groupby (str, list): Name(s) of column(s) defining the grouping.
            args, kwargs: Optional positional and keyword arguments to pass
                onto the function call.
        '''
        grouper = self.get_grouper(groupby)
        return self.values.groupby(grouper).apply(func, *args, **kwargs)


class SimpleColumn(BIDSColumn):
    ''' Represents a simple design matrix column that has no timing
    information.
    Args:
        name (str): Name of the column.
        data (DataFrame): A pandas DataFrame minimally containing a column
            named 'amplitude' as well as any identifying entities.
        factor_name (str): If this column is derived from a categorical factor
            (e.g., level 'A' in a 'trial_type' column), the name of the
            originating factor.
        level_index (int): The positional index of the current level in the
            originating categorical factor. Ignored if factor_name is None.
        level_name (str): The name of the current level in the originating
            categorical factor, if applicable.
    '''

    # Columns that define special properties (e.g., onset, duration). These
    # will be stored separately from the main data object, and are accessible
    # as properties on the SimpleColumn instance.
    _property_columns = set()
    _entity_columns = {'condition', 'amplitude', 'factor'}

    def __init__(self, name, data, factor_name=None,
                 level_index=None, level_name=None):

        self.factor_name = factor_name
        self.level_index = level_index
        self.level_name = level_name

        for sc in self._property_columns:
            setattr(self, sc, data[sc].values)

        ent_cols = list(set(data.columns) - self._entity_columns -
                        self._property_columns)
        self.entities = data.loc[:, ent_cols]

        values = data['amplitude'].reset_index(drop=True)
        values.name = name

        super(SimpleColumn, self).__init__(name, values)

    def aggregate(self, unit, func='mean'):

        levels = ['run', 'session', 'subject']
        groupby = set(levels[levels.index(unit):]) & set(self.entities.columns)
        groupby = list(groupby)

        entities = self.entities.loc[:, groupby].reset_index(drop=True)
        values = pd.DataFrame({'amplitude': self.values.values})
        data = pd.concat([values, entities], axis=1)
        data = data.groupby(groupby, as_index=False).agg(func)
        return SimpleColumn(self.name, data, self.factor_name,
                            self.level_index, self.level_name)

    def to_df(self, condition=True, entities=True):
        ''' Convert to a DataFrame, with columns for name and entities.
        Args:
            condition (bool): If True, adds a column for condition name, and
                names the amplitude column 'amplitude'. If False, returns just
                onset, duration, and amplitude, and gives the amplitude column
                the current column name.
            entities (bool): If True, adds extra columns for all entities.
        '''
        amp = 'amplitude' if condition else self.name
        data = pd.DataFrame({amp: self.values.values.ravel()})

        for sc in self._property_columns:
            data[sc] = getattr(self, sc)

        if condition:
            data['condition'] = self.name

        if entities:
            ent_data = self.entities.reset_index(drop=True)
            data = pd.concat([data, ent_data], axis=1)

        return data

    def split(self, grouper):
        ''' Split the current SparseEventColumn into multiple columns.
        Args:
            grouper (iterable): list to groupby, where each unique value will
                be taken as the name of the resulting column.
        Returns:
            A list of SparseEventColumns, one per unique value in the grouper.
        '''
        data = self.to_df(condition=True, entities=False)
        data = data.drop('condition', axis=1)
        # data = pd.DataFrame(dict(onset=self.onset, duration=self.duration,
        #                          amplitude=self.values.values))
        # data = pd.concat([data, self.index.reset_index(drop=True)], axis=1)

        subsets = []
        for i, (name, g) in enumerate(data.groupby(grouper)):
            name = '%s.%s' % (self.name, name)
            col = self.__class__(name, g, level_name=name,
                                 factor_name=self.name, level_index=i)
            subsets.append(col)
        return subsets

    @property
    def index(self):
        ''' An index of all named entities. '''
        return self.entities


class SparseEventColumn(SimpleColumn):
    ''' A sparse representation of a single column of events.
    Args:
        name (str): Name of the column.
        data (DataFrame): A pandas DataFrame minimally containing the columns
            'onset', 'duration', and 'amplitude'.
        factor_name (str): If this column is derived from a categorical factor
            (e.g., level 'A' in a 'trial_type' column), the name of the
            originating factor.
        level_index (int): The positional index of the current level in the
            originating categorical factor. Ignored if factor_name is None.
        level_name (str): The name of the current level in the originating
            categorical factor, if applicable.
    '''

    _property_columns = {'onset', 'duration'}

    def to_dense(self, sampling_rate=None):
        ''' Convert the current sparse column to a dense representation.
        Returns: A DenseEventColumn. '''
        if sampling_rate is None:
            sampling_rate = self.collection.sampling_rate
        duration = int(sampling_rate * len(self.collection.dense_index) /
                       self.collection.sampling_rate)
        ts = np.zeros(duration)

        onsets = np.ceil(self.onset * sampling_rate).astype(int)
        durations = np.round(self.duration * sampling_rate).astype(int)

        for i, row in enumerate(self.values.values):
            file_id = self.entities['unique_run_id'].values[i]
            run_onset = self.collection.run_infos[file_id].start
            ev_start = onsets[i] + int(math.ceil(run_onset * sampling_rate))
            ev_end = ev_start + durations[i]
            ts[ev_start:ev_end] = row

        ts = pd.DataFrame(ts)

        return DenseEventColumn(self.name, ts)


class DenseEventColumn(BIDSColumn):
    ''' A dense representation of a single column.

    Args:
        name (str): The name of the column
        values (NDArray): The values/amplitudes to store
        sampling_rate (float): Optional sampling rate (in Hz) to use. Must
            match the sampling rate used to generate the values. If None,
            the collection's sampling rate will be used.
    '''

    def __init__(self, name, values, sampling_rate):
        self.sampling_rate = sampling_rate
        super(DenseEventColumn, self).__init__(name, values)
        self._index = _build_dense_index(sampling_rate)

    @property
    def index(self):
        ''' An index of all named entities. '''
        return self._index

    def split(self, grouper):
        ''' Split the current DenseEventColumn into multiple columns.
        Args:
            grouper (DataFrame): binary DF specifying the design matrix to
                use for splitting. Number of rows must match current
                DenseEventColumn; a new DenseEventColumn will be generated
                for each column in the grouper.
        Returns:
            A list of DenseEventColumns, one per unique value in the grouper.
        '''
        df = grouper * self.values
        names = df.columns
        return [DenseEventColumn('%s.%s' % (self.name, name),
                                 df[name].values)
                for i, name in enumerate(names)]

    def aggregate(self, unit, func='mean'):

        levels = ['run', 'session', 'subject']
        groupby = set(levels[levels.index(unit):]) & \
            set(self.index.columns)
        groupby = list(groupby)

        entities = self._index.loc[:, groupby].reset_index(drop=True)
        values = pd.DataFrame({'amplitude': self.values.values.ravel()})
        data = pd.concat([values, entities], axis=1)
        data = data.groupby(groupby, as_index=False).agg(func)
        return SimpleColumn(self.name, data)

    def resample(self, sampling_rate, kind='linear'):
        ''' Resample the column to the specified sampling rate.

        Args:
            sampling_rate (int, float): Target sampling rate (in Hz)
            kind (str): Argument to pass to scipy's interp1d; indicates the
                kind of interpolation approach to use. See interp1d docs for
                valid values.
        '''

        if sampling_rate == self.sampling_rate:
            return

        # Use the collection's index if possible
        if sampling_rate == self.collection.sampling_rate:
            self._index = self.collection.index
            return

        old_sr = self.sampling_rate
        n = len(self.index)

        self._index = _build_dense_index(sampling_rate)

        x = np.arange(n)
        num = int(np.ceil(n * sampling_rate / old_sr))

        f = interp1d(x, self.values.values.ravel(), kind=kind)
        x_new = np.linspace(0, n - 1, num=num)
        self.values = pd.DataFrame(f(x_new))

        self.sampling_rate = sampling_rate
