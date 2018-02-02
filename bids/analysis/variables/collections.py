import pandas as pd
from copy import copy
from pandas.api.types import is_numeric_dtype
import warnings
import re
from .variables import SparseRunVariable, SimpleVariable, DenseRunVariable
from .utils import _build_dense_index


class BIDSVariableCollection(object):

    ''' A container for one or more variables extracted from variable files
    at a single level of analysis defined in the BIDS spec (i.e., 'run',
    'session', 'subject', or 'dataset').
    Args:
        unit (str): The unit of analysis. Each row in the stored column(s)
            is taken to reflect a single unit. Must be one of 'time', 'run',
            'session', 'subject', or 'dataset'.
        entities (list): A list of entities defined for all variables in this
            collection.
    '''

    def __init__(self, unit, entities):

        self.unit = unit
        self.entities = entities
        self.columns = {}
        self.dense_index = None

    def merge_columns(self, columns=None):
        ''' Merge columns into one DF.
        Args:
            columns (list): Optional list of column names to retain; if None,
                all columns are written out.
        Returns: A pandas DataFrame.
        '''

        _cols = self.columns.values()

        # Retain only specific columns if desired
        if columns is not None:
            _cols = [c for c in _cols if c.name in columns]

        return pd.concat([c.to_df() for c in _cols], axis=0)

    def clone(self):
        ''' Returns a shallow copy of the current instance, except that all
        columns are deep-cloned.
        '''
        clone = copy(self)
        clone.columns = {k: v.clone() for (k, v) in self.columns.items()}
        return clone

    def aggregate(self, unit, agg_func='mean', categorical_agg_func=None):
        ''' Aggregate variable values from a lower level at a higher level.
        Args:
            unit (str): The unit of aggregation. The returned collection will
                have one row per value of this unit.
            agg_func (str, Callable): Aggregation function to use. Must be
                either a named function recognized by apply() in pandas, or a
                Callable that takes a DataFrame and returns a Series or
                DataFrame.
            categorical_agg_func (str, Callable): Aggregation function to use
                for categorical variables. Must be a function that returns
                valid output given categorical inputs. If None, aggregation
                will only proceed if all categorical columns have exactly one
                unique value.
        '''

        for col in self.columns.values():
            if is_numeric_dtype(col.values):
                _func = agg_func
            else:
                if categorical_agg_func is not None:
                    _func = categorical_agg_func
                elif col.values.nunique() > 1:
                    msg = ("Column %s is categorical and has more than one "
                           "unique value. You must explicitly specify an "
                           "aggregation function in the categorical_agg_func "
                           "argument.")
                    raise ValueError(msg)
                else:
                    _func = 'first'

            self[col.name] = col.aggregate(unit, _func)

    def __getitem__(self, col):
        return self.columns[col]

    def __setitem__(self, col, obj):
        # Ensure name matches collection key, but raise warning if needed.
        if obj.name != col:
            warnings.warn("The provided key to use in the collection ('%s') "
                          "does not match the passed Column object's existing "
                          "name ('%s'). The Column name will be set to match "
                          "the provided key." % (col, obj.name))
            obj.name = col
        self.columns[col] = obj

    def match_columns(self, pattern, return_type='name'):
        ''' Return columns whose names match the provided regex pattern.
        Args:
            pattern (str): A regex pattern to match all column names against.
            return_type (str): What to return. Must be one of:
                'name': Returns a list of names of matching columns.
                'column': Returns a list of Column objects whose names match.
        '''
        pattern = re.compile(pattern)
        cols = [c for c in self.columns.values() if pattern.search(c.name)]
        return cols if return_type.startswith('col') \
            else [c.name for c in cols]

    def _construct_design_matrix(self, data, groupby=None, aggregate=None,
                                 add_intercept=None, drop_entities=False,
                                 drop_cols=None, **kwargs):

        # subset the data if needed
        if kwargs:
            # TODO: make sure this handles constraints on int columns properly
            bad_keys = list(set(kwargs.keys()) - set(data.columns))
            if bad_keys:
                raise ValueError("The following query constraints do not map "
                                 "onto existing columns: %s." % bad_keys)
            query = ' and '.join(["{} in {}".format(k, v)
                                  for k, v in kwargs.items()])
            data = data.query(query)

        if groupby and aggregate is not None:
            groupby = list(set(groupby) & set(data.columns))
            data = data.groupby(groupby).agg(aggregate).reset_index()

        if add_intercept:
            data.insert(0, 'intercept', 1)

        if drop_cols is None:
            drop_cols = []

        # Always drop columns meant for internal use
        drop_cols += ['unique_run_id', 'time']

        # Optionally drop entities
        if drop_entities:
            drop_cols += self.entities

        drop_cols = list(set(drop_cols) & set(data.columns))
        data = data.drop(drop_cols, axis=1).reset_index(drop=True)
        return data

    def get_design_matrix(self, columns=None, groupby=None, aggregate=None,
                          add_intercept=False, drop_entities=False, **kwargs):
        ''' Returns a design matrix constructed by combining the current
        BIDSVariableCollection's columns.
        Args:
            columns (list): Optional list naming columns to include in the
                design matrix. If None (default), all columns are included.
            groupby (str, list): Optional name (or list of names) of design
                variables to group by.
            aggregate (str, Callable): Optional aggregation function to apply
                to groups if groupby is not None (ignored otherwise). Must be
                either a named function recognized by pandas, or a Callable
                that takes a DataFrame and returns a Series or DataFrame.
            add_intercept (bool): If True, adds an intercept column (i.e.,
                constant column of 1's) to the returned design matrix.
            drop_entities (bool): If True, entities are stripped from the
                returned design matrix. When False, entities like 'subject' and
                'run' are included as columns in the matrix, leaving it up to
                the user to remove them if appropriate.
            kwargs: Optional query constraints understood by
                pd.DataFrame.query(). Must be compatible with the 'in' syntax;
                e.g., passing subject=['01', '02', '03'] would return only
                rows that match the first 3 subjects.

        Returns: A pandas DataFrame, where units of analysis (seconds, runs,
            etc.) are in rows, and variables/columns are in columns.

        '''

        if columns is None:
            columns = list(self.columns.keys())

        if groupby is None:
            groupby = []

        data = self.merge_columns(columns=columns)

        return self._construct_design_matrix(data, groupby, aggregate,
                                             add_intercept, drop_entities,
                                             **kwargs)


class BIDSRunVariableCollection(BIDSVariableCollection):

    ''' A container for one or more EventColumns--i.e., Columns that have a
    temporal dimension.

    Args:
        unit (str): The unit of analysis. Each row in the stored column(s)
            is taken to reflect a single unit. Must be one of 'time', 'run',
            'session', 'subject', or 'dataset'.
        entities (list): A list of entities defined for all variables in this
            collection.
        sampling_rate (float): Sampling rate (in Hz) to use when working with
            dense representations of variables.
        repetition_time (float): TR of corresponding image(s) in seconds.
    '''

    def __init__(self, unit, entities, sampling_rate=None,
                 repetition_time=None):

        self.sampling_rate = sampling_rate
        self.repetition_time = repetition_time
        self.run_infos = []
        self.dense_index = None
        super(BIDSRunVariableCollection, self).__init__(unit, entities)

    def _get_sampling_rate(self, sr):
        return self.repetition_time if sr == 'tr' else sr

    def _build_dense_index(self):
        ''' Build an index of all tracked entities for all dense columns. '''

        if not self.run_infos:
            return

        self.dense_index = _build_dense_index(self.run_infos,
                                              self.sampling_rate)

    def _none_dense(self):
        return all([isinstance(c, SimpleVariable)
                    for c in self.columns.values()])

    def _all_dense(self):
        return all([isinstance(c, DenseRunVariable)
                    for c in self.columns.values()])

    @property
    def index(self):
        return self.dense_index

    def clone(self):
        ''' Returns a shallow copy of the current instance, except that all
        columns and the index are deep-cloned.
        '''
        clone = super(BIDSRunVariableCollection, self).clone()
        clone.dense_index = self.dense_index.copy()
        return clone

    def resample(self, sampling_rate='tr', force_dense=False, in_place=False,
                 kind='linear'):
        ''' Resample all dense columns (and optionally, sparse ones) to the
        specified sampling rate.

        Args:
            sampling_rate (int, float): Target sampling rate (in Hz)
            force_dense (bool): if True, all sparse columns will be forced to
                dense.
            in_place (bool): When True, all columns are overwritten in-place.
                When False, returns resampled versions of all columns.
            kind (str): Argument to pass to scipy's interp1d; indicates the
                kind of interpolation approach to use. See interp1d docs for
                valid values.
        '''

        # Store old sampling rate-based variables
        sampling_rate = self._get_sampling_rate(sampling_rate)

        columns = {}

        for name, col in self.columns.items():
            if isinstance(col, SparseRunVariable):
                if force_dense and is_numeric_dtype(col.values):
                    columns[name] = col.to_dense(sampling_rate)
            else:
                col = col.clone()
                col.resample(sampling_rate, kind)
                columns[name] = col

        if in_place:
            for k, v in columns.items():
                self.columns[k] = v
            # Rebuild the dense index
            self.sampling_rate = sampling_rate
            self._build_dense_index()

        else:
            return columns

    def merge_columns(self, columns=None, sparse=True, sampling_rate='tr'):
        ''' Merge columns into one DF.
        Args:
            columns (list): Optional list of column names to retain; if None,
                all columns are written out.
            sparse (bool): If True, columns will be kept in a sparse format
                provided they are all internally represented as such. If False,
                a dense matrix (i.e., uniform sampling rate for all events)
                will be exported. Will be ignored if at least one column is
                dense.
            sampling_rate (float): If a dense matrix is written out, the
                sampling rate (in Hz) to use for downsampling. Defaults to the
                value currently set in the instance.
        Returns: A pandas DataFrame.
        '''

        if sparse and self._none_dense():
            return super(BIDSRunVariableCollection,
                         self).merge_columns(columns)

        sampling_rate = self._get_sampling_rate(sampling_rate)

        # Make sure all columns have the same sampling rate
        _cols = self.resample(sampling_rate, force_dense=True,
                              in_place=False).values()

        # Retain only specific columns if desired
        if columns is not None:
            _cols = [c for c in _cols if c.name in columns]

        _cols = [c for c in _cols if c.name not in ["event_file_id", "time"]]

        # Merge all data into one DF
        dfs = [pd.Series(c.values.iloc[:, 0], name=c.name) for c in _cols]
        # Convert datetime to seconds and add duration column
        dense_index = self.dense_index.copy()
        onsets = self.dense_index.pop('time').values.astype(float) / 1e+9
        timing = pd.DataFrame({'onset': onsets})
        timing['duration'] = 1. / sampling_rate
        dfs = [timing] + dfs + [dense_index]
        data = pd.concat(dfs, axis=1)

        return data

    def get_design_matrix(self, columns=None, groupby=None, aggregate=None,
                          add_intercept=False, drop_entities=False,
                          drop_timing=False, sampling_rate='tr', **kwargs):
        ''' Returns a design matrix constructed by combining the current
        BIDSVariableCollection's columns.
        Args:
            columns (list): Optional list naming columns to include in the
                design matrix. If None (default), all columns are included.
            groupby (str, list): Optional name (or list of names) of design
                variables to group by.
            aggregate (str, Callable): Optional aggregation function to apply
                to groups if groupby is not None (ignored otherwise). Must be
                either a named function recognized by pandas, or a Callable
                that takes a DataFrame and returns a Series or DataFrame.
            add_intercept (bool): If True, adds an intercept column (i.e.,
                constant column of 1's) to the returned design matrix.
            drop_entities (bool): If True, entities are stripped from the
                returned design matrix. When False, entities like 'subject' and
                'run' are included as columns in the matrix, leaving it up to
                the user to remove them if appropriate.
            sampling_rate (float, str): The sampling rate to use for the
                returned design matrix. Value must be either a float
                expressed in seconds, or the special value 'tr', which
                uses the associated scan's repetition time as the sampling
                rate.
            kwargs: Optional query constraints understood by
                pd.DataFrame.query(). Must be compatible with the 'in' syntax;
                e.g., passing subject=['01', '02', '03'] would return only
                rows that match the first 3 subjects.

        Returns: A pandas DataFrame, where units of analysis (seconds, runs,
            etc.) are in rows, and variables/columns are in columns.

        '''
        if columns is None:
            columns = list(self.columns.keys())

        if groupby is None:
            groupby = []

        data = self.merge_columns(columns=columns, sampling_rate=sampling_rate,
                                  sparse=True)

        return self._construct_design_matrix(data, groupby, aggregate,
                                             add_intercept, drop_entities,
                                             **kwargs)
