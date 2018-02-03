import pandas as pd
from copy import copy
from pandas.api.types import is_numeric_dtype
import warnings
import re
from .variables import (SparseRunVariable, SimpleVariable, DenseRunVariable,
                        merge_variables)
from collections import defaultdict


class BIDSVariableCollection(object):

    ''' A container for one or more variables extracted from variable files
    at a single level of analysis.

    Args:
        level (str): The level of analysis. Each row in the stored column(s)
            is taken to reflect a single level. Must be one of 'run',
            'session', 'subject', or 'dataset'.
        variables (list): A list of Variables.
    '''

    def __init__(self, variables):

        SOURCE_TO_LEVEL = {
            'events': 'run',
            'physio': 'run',
            'stim': 'run',
            'confounds': 'run',
            'scans': 'session',
            'sessions': 'subject',
            'participants': 'dataset'
        }

        var_levels = set([SOURCE_TO_LEVEL[v.source] for v in variables])

        # TODO: relax this requirement & allow implicit merging between levels
        if len(var_levels) > 1:
            raise ValueError("A Collection cannot be initialized from "
                             "variables at more than one level of analysis. "
                             "Levels found in input variables: %s" %
                             var_levels)

        self.level = list(var_levels)[0]
        variables = self.merge_variables(variables)
        self.variables = {v.name: v for v in variables}

    @staticmethod
    def merge_variables(variables):
        # Concatenates Variables along row axis (i.e., in cases where the same
        # Variable has multiple replicates for different Runs, Sessions, etc.)
        var_dict = defaultdict(list)
        for v in variables:
            var_dict[v.name].append(v)
        return [merge_variables(vars_) for vars_ in list(var_dict.values())]

    def to_df(self, variables=None, format='wide'):
        ''' Merge variables into a single pandas DataFrame.

        Args:
            variables (list): Optional list of column names to retain; if None,
                all variables are returned.
            format (str): Whether to return a DataFrame in 'wide' or 'long'
                format.

        Returns: A pandas DataFrame.
        '''

        _vars = self.variables
        if variables is not None:
            _vars = [v for v in variables if v.name in variables]

        # _vars = self.variables.values()

        # Need to index by entities

        return pd.concat([v.to_df() for v in _vars], axis=1)

    def clone(self):
        ''' Returns a shallow copy of the current instance, except that all
        variables are deep-cloned.
        '''
        clone = copy(self)
        clone.variables = {k: v.clone() for (k, v) in self.variables.items()}
        return clone

    def aggregate(self, level, agg_func='mean', categorical_agg_func=None):
        ''' Aggregate variable values from a lower level at a higher level.

        Args:
            level (str): The level of aggregation. The returned collection will
                have one row per value of this level.
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

        for var in self.variables.values():
            if is_numeric_dtype(var.values):
                _func = agg_func
            else:
                if categorical_agg_func is not None:
                    _func = categorical_agg_func
                elif var.values.nunique() > 1:
                    msg = ("Column %s is categorical and has more than one "
                           "unique value. You must explicitly specify an "
                           "aggregation function in the categorical_agg_func "
                           "argument.")
                    raise ValueError(msg)
                else:
                    _func = 'first'

            self[var.name] = var.aggregate(level, _func)

    def __getitem__(self, var):
        return self.variables[var]

    def __setitem__(self, var, obj):
        # Ensure name matches collection key, but raise warning if needed.
        if obj.name != var:
            warnings.warn("The provided key to use in the collection ('%s') "
                          "does not match the passed Column object's existing "
                          "name ('%s'). The Column name will be set to match "
                          "the provided key." % (var, obj.name))
            obj.name = var
        self.variables[var] = obj

    def match_variables(self, pattern, return_type='name'):
        ''' Return columns whose names match the provided regex pattern.

        Args:
            pattern (str): A regex pattern to match all variable names against.
            return_type (str): What to return. Must be one of:
                'name': Returns a list of names of matching variables.
                'variable': Returns a list of Variable objects whose names
                match.
        '''
        pattern = re.compile(pattern)
        vars_ = [v for v in self.variables.values() if pattern.search(v.name)]
        return vars_ if return_type.startswith('var') \
            else [v.name for v in vars_]

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

    def get_design_matrix(self, variables=None, groupby=None, aggregate=None,
                          add_intercept=False, drop_entities=False, **kwargs):
        ''' Returns a design matrix constructed by combining the current
        BIDSVariableCollection's variables.

        Args:
            variables (list): Optional list naming variables to include in the
                design matrix. If None (default), all variables are included.
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

        Returns: A pandas DataFrame, where levels of analysis (seconds, runs,
            etc.) are in rows, and variables/columns are in columns.

        '''

        if variables is None:
            variables = list(self.variables.keys())

        if groupby is None:
            groupby = []

        data = self.merge_columns(variables=variables)

        return self._construct_design_matrix(data, groupby, aggregate,
                                             add_intercept, drop_entities,
                                             **kwargs)


class BIDSRunVariableCollection(BIDSVariableCollection):

    ''' A container for one or more RunVariables--i.e., Variables that have a
    temporal dimension.

    Args:
        variables (list): A list of Variables.
        sampling_rate (float): Sampling rate (in Hz) to use when working with
            dense representations of variables. If None, defaults to 10.
    '''

    def __init__(self, variables, sampling_rate=None):
        # Don't put the default value in signature b/c None is passed from
        # several places and we don't want multiple conflicting defaults.
        self.sampling_rate = sampling_rate or 10
        super(BIDSRunVariableCollection, self).__init__(variables)

    def _none_dense(self):
        return all([isinstance(v, SimpleVariable)
                    for v in self.variables.values()])

    def _all_dense(self):
        return all([isinstance(v, DenseRunVariable)
                    for v in self.variables.values()])

    def resample(self, sampling_rate=None, force_dense=False, in_place=False,
                 kind='linear'):
        ''' Resample all dense variables (and optionally, sparse ones) to the
        specified sampling rate.

        Args:
            sampling_rate (int, float): Target sampling rate (in Hz). If None,
                uses the instance sampling rate.
            force_dense (bool): if True, all sparse variables will be forced to
                dense.
            in_place (bool): When True, all variables are overwritten in-place.
                When False, returns resampled versions of all variables.
            kind (str): Argument to pass to scipy's interp1d; indicates the
                kind of interpolation approach to use. See interp1d docs for
                valid values.
        '''

        # Store old sampling rate-based variables
        sampling_rate = sampling_rate or self.sampling_rate

        variables = {}

        for name, var in self.variables.items():
            if isinstance(var, SparseRunVariable):
                if force_dense and is_numeric_dtype(var.values):
                    variables[name] = var.to_dense(sampling_rate)
            else:
                variables[name] = var.resample(sampling_rate, kind)
                variables[name] = var

        if in_place:
            for k, v in variables.items():
                self.variables[k] = v
            self.sampling_rate = sampling_rate
        else:
            return variables

    def to_df(self, columns=None, sparse=True, sampling_rate=None):
        ''' Merge columns into a single pandas DataFrame.

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

        sampling_rate = sampling_rate or self.sampling_rate

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
                          drop_timing=False, sampling_rate=None, **kwargs):
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

        Returns: A pandas DataFrame, where levels of analysis (seconds, runs,
            etc.) are in rows, and variables/columns are in columns.

        '''
        if columns is None:
            columns = list(self.variables.keys())

        if groupby is None:
            groupby = []

        data = self.to_df(columns=columns, sampling_rate=sampling_rate,
                          sparse=True)

        return self._construct_design_matrix(data, groupby, aggregate,
                                             add_intercept, drop_entities,
                                             **kwargs)
