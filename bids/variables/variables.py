import numpy as np
import pandas as pd
import math
from copy import deepcopy
from abc import abstractmethod, abstractclassmethod, ABCMeta
from scipy.interpolate import interp1d
from bids.utils import listify
from itertools import chain
from six import add_metaclass


@add_metaclass(ABCMeta)
class BIDSVariable(object):
    ''' Base representation of a column in a BIDS project. '''

    # Columns that define special properties (e.g., onset, duration). These
    # will be stored separately from the main data object, and are accessible
    # as properties on the BIDSVariable instance.
    _property_columns = set()

    def __init__(self, name, values, source):
        self.name = name
        self.values = values
        self.source = source

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
            result.values = pd.DataFrame(data)

        if kwargs:
            for k, v in kwargs.items():
                setattr(result, k, v)

        # Need to update name on Series as well
        # result.values.name = kwargs.get('name', self.name)
        return result

    @abstractmethod
    def aggregate(self, unit, level, func):
        pass

    @classmethod
    def merge(cls, variables, name=None, **kwargs):

        variables = listify(variables)
        if len(variables) == 1:
            return variables[0]

        var_names = set([v.name for v in variables])
        if len(var_names) > 1:
            raise ValueError("Columns with different names cannot be merged. "
                             "Column names provided: %s" % var_names)

        if name is None:
            name = variables[0].name

        return cls._merge(variables, name, **kwargs)

    @abstractclassmethod
    def _merge(cls, variables, name, **kwargs):
        pass

    def get_grouper(self, groupby='run'):
        ''' Return a pandas Grouper object suitable for use in groupby calls.
        Args:
            groupby (str, list): Name(s) of column(s) defining the grouper
                object. Anything that would be valid inside a .groupby() call
                on a pandas structure.
        Returns:
            A pandas Grouper object constructed from the specified columns
                of the current index.
        '''
        return pd.core.groupby._get_grouper(self.entities, groupby)[0]

    def apply(self, func, groupby='run', *args, **kwargs):
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


class SimpleVariable(BIDSVariable):
    ''' Represents a simple design matrix column that has no timing
    information.

    Args:
        name (str): Name of the column.
        data (DataFrame): A pandas DataFrame minimally containing a column
            named 'amplitude' as well as any identifying entities.
    '''

    _entity_columns = {'condition', 'amplitude'}

    def __init__(self, name, data, source, **kwargs):

        ent_cols = list(set(data.columns) - self._entity_columns)
        self.entities = data.loc[:, ent_cols]

        values = data['amplitude'].reset_index(drop=True)
        values.name = name

        super(SimpleVariable, self).__init__(name, values, source)

    def aggregate(self, unit, func='mean'):

        levels = ['task', 'run', 'session', 'subject']
        groupby = set(levels[levels.index(unit):]) & set(self.entities.columns)
        groupby = list(groupby)

        entities = self.entities.loc[:, groupby].reset_index(drop=True)
        values = pd.DataFrame({'amplitude': self.values.values})
        data = pd.concat([values, entities], axis=1)
        data = data.groupby(groupby, as_index=False).agg(func)
        return SimpleVariable(self.name, data)

    def split(self, grouper):
        ''' Split the current SparseRunVariable into multiple columns.
        Args:
            grouper (iterable): list to groupby, where each unique value will
                be taken as the name of the resulting column.
        Returns:
            A list of SparseRunVariables, one per unique value in the
            grouper.
        '''
        data = self.to_df(condition=True, entities=False)
        data = data.drop('condition', axis=1)

        subsets = []
        for i, (name, g) in enumerate(data.groupby(grouper)):
            name = '%s.%s' % (self.name, name)
            args = [name, g, self.source]
            if hasattr(self, 'run_info'):
                args.append(self.run_info)
            col = self.__class__(*args)
            subsets.append(col)
        return subsets

    @classmethod
    def _merge(cls, variables, name, **kwargs):
        dfs = [v.to_df() for v in variables]
        data = pd.concat(dfs, axis=0).reset_index(drop=True)
        data = data.rename(columns={name: 'amplitude'})
        return cls(name, data, source=variables[0].source, **kwargs)


class SparseRunVariable(SimpleVariable):
    ''' A sparse representation of a single column of events.

    Args:
        name (str): Name of the column.
        data (DataFrame): A pandas DataFrame minimally containing the columns
            'onset', 'duration', and 'amplitude'.
        durations (float, list): ???
    '''

    _property_columns = {'onset', 'duration'}

    def __init__(self, name, data, run_info, source, **kwargs):
        if hasattr(run_info, 'duration'):
            run_info = [run_info]
        self.run_info = run_info
        for sc in self._property_columns:
            setattr(self, sc, data.pop(sc).values)
        super(SparseRunVariable, self).__init__(name, data, source, **kwargs)

    def get_duration(self):
        return sum([r.duration for r in self.run_info])

    def to_dense(self, sampling_rate):
        ''' Convert the current sparse column to a dense representation.
        Returns: A DenseRunVariable. '''
        duration = math.ceil(sampling_rate * self.get_duration())
        ts = np.zeros(duration, dtype=self.values.dtype)

        onsets = np.round(self.onset * sampling_rate).astype(int)
        durations = np.round(self.duration * sampling_rate).astype(int)

        run_i, start, last_ind = 0, 0, 0
        for i, val in enumerate(self.values.values):
            if onsets[i] < last_ind:
                start += self.run_info[run_i].duration * sampling_rate
                run_i += 1
            _onset = start + onsets[i]
            _offset = _onset + durations[i]
            ts[_onset:_offset] = val
            last_ind = onsets[i]

        run_info = self.run_info.copy()
        return DenseRunVariable(self.name, ts, run_info, self.source,
                                sampling_rate)

    @classmethod
    def _merge(cls, variables, name, **kwargs):
        run_info = list(chain(*[v.run_info for v in variables]))
        return super(SparseRunVariable, cls)._merge(variables, name,
                                                    run_info=run_info,
                                                    **kwargs)


class DenseRunVariable(BIDSVariable):
    ''' A dense representation of a single column.

    Args:
        name (str): The name of the column
        values (NDArray): The values/amplitudes to store
        sampling_rate (float): Optional sampling rate (in Hz) to use. Must
            match the sampling rate used to generate the values. If None,
            the collection's sampling rate will be used.
    '''

    def __init__(self, name, values, run_info, source, sampling_rate):

        values = pd.DataFrame(values)

        super(DenseRunVariable, self).__init__(name, values, source)

        if hasattr(run_info, 'duration'):
            run_info = [run_info]
        self.run_info = run_info
        self.sampling_rate = sampling_rate
        self.entities = self.build_entity_index(run_info, sampling_rate)

    def split(self, grouper):
        ''' Split the current DenseRunVariable into multiple columns.
        Args:
            grouper (DataFrame): binary DF specifying the design matrix to
                use for splitting. Number of rows must match current
                DenseRunVariable; a new DenseRunVariable will be generated
                for each column in the grouper.
        Returns:
            A list of DenseRunVariables, one per unique value in the grouper.
        '''
        values = grouper.values * self.values.values
        df = pd.DataFrame(values, columns=grouper.columns)
        return [DenseRunVariable('%s.%s' % (self.name, name), df[name].values,
                                 self.run_info, self.source,
                                 self.sampling_rate)
                for i, name in enumerate(df.columns)]

    def aggregate(self, unit, func='mean'):

        levels = ['task', 'run', 'session', 'subject']
        groupby = set(levels[levels.index(unit):]) & \
            set(self.entities.columns)
        groupby = list(groupby)

        entities = self._index.loc[:, groupby].reset_index(drop=True)
        values = pd.DataFrame({'amplitude': self.values.values.ravel()})
        data = pd.concat([values, entities], axis=1)
        data = data.groupby(groupby, as_index=False).agg(func)
        return SimpleVariable(self.name, data)

    @staticmethod
    def build_entity_index(run_info, sampling_rate):
        index = []
        sr = int(round(1000. / sampling_rate))
        for run in run_info:
            reps = int(math.ceil(run.duration * sampling_rate))
            ent_vals = list(run.entities.values())
            data = np.broadcast_to(ent_vals, (reps, len(ent_vals)))
            df = pd.DataFrame(data, columns=list(run.entities.keys()))
            df['time'] = pd.date_range(0, periods=len(df), freq='%sms' % sr)
            index.append(df)
        return pd.concat(index, axis=0).reset_index(drop=True)

    def resample(self, sampling_rate, inplace=False, kind='linear'):
        ''' Resample the Variable to the specified sampling rate.
        Args:
            sampling_rate (int, float): Target sampling rate (in Hz)
            inplace (bool): If True, performs resampling in-place. If False,
                returns a resampled copy of the current Variable.
            kind (str): Argument to pass to scipy's interp1d; indicates the
                kind of interpolation approach to use. See interp1d docs for
                valid values.
        '''
        if not inplace:
            var = self.clone()
            var.resample(sampling_rate, True, kind)
            return var

        if sampling_rate == self.sampling_rate:
            return

        old_sr = self.sampling_rate
        n = len(self.entities)

        self.entities = self.build_entity_index(self.run_info, sampling_rate)

        x = np.arange(n)
        num = int(np.ceil(n * sampling_rate / old_sr))

        f = interp1d(x, self.values.values.ravel(), kind=kind)
        x_new = np.linspace(0, n - 1, num=num)
        self.values = pd.DataFrame(f(x_new))

        self.sampling_rate = sampling_rate

    def to_df(self, condition=True, entities=True, timing=True):
        ''' Convert to a DataFrame, with columns for name and entities.
        Args:
            condition (bool): If True, adds a column for condition name, and
                names the amplitude column 'amplitude'. If False, returns just
                onset, duration, and amplitude, and gives the amplitude column
                the current column name.
            entities (bool): If True, adds extra columns for all entities.
            timing (bool): If True, includes onset and duration columns (even
                though events are sampled uniformly). If False, omits them.
        '''
        df = super(DenseRunVariable, self).to_df(condition, entities)

        if timing:
            df['onset'] = df['time'].values.astype(float) / 1e+9
            df['duration'] = 1. / self.sampling_rate

        return df.drop('time', axis=1)

    @classmethod
    def _merge(cls, variables, name, sampling_rate=None, **kwargs):

        if not isinstance(sampling_rate, int):
            rates = set([v.sampling_rate for v in variables])
            if len(rates) == 1:
                sampling_rate = list(rates)[0]
            else:
                if sampling_rate is 'auto':
                    sampling_rate = max(rates)
                else:
                    msg = ("Cannot merge DenseRunVariables (%s) with different"
                           " sampling rates (%s). Either specify an integer "
                           "sampling rate to use for all variables, or set "
                           "sampling_rate='auto' to use the highest sampling "
                           "rate found." % (name, rates))
                    raise ValueError(msg)

        variables = [v.resample(sampling_rate) for v in variables]
        values = pd.concat([v.values for v in variables], axis=0)
        run_info = list(chain(*[v.run_info for v in variables]))
        source = variables[0].source
        return DenseRunVariable(name, values, run_info, source, sampling_rate)


def merge_variables(variables, **kwargs):

    classes = set([v.__class__ for v in variables])
    if len(classes) > 1:
        raise ValueError("Variables of different classes cannot be merged. "
                         "Variables passed are of classes: %s" % classes)

    sources = set([v.source for v in variables])
    if len(sources) > 1:
        raise ValueError("Variables extracted from different types of files "
                         "cannot be merged. Sources found: %s" % sources)

    return list(classes)[0].merge(variables, **kwargs)
