import pandas as pd
from copy import copy
from pandas.api.types import is_numeric_dtype
import warnings
import re
from .variables import (SparseRunVariable, SimpleVariable, DenseRunVariable,
                        merge_variables, BIDSVariable)
from collections import defaultdict
from itertools import chain
from bids.utils import listify


class BIDSVariableCollection(object):

    ''' A container for one or more variables extracted from variable files
    at a single level of analysis.

    Args:
        variables (list): A list of BIDSVariables or SimpleVariables.

    Notes:
        * Variables in the list must all share the same analysis level, which
          must be one of 'session', 'subject', or 'dataset' level. For
          run-level Variables, use the BIDSRunVariableCollection.
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
    def merge_variables(variables, **kwargs):
        ''' Concatenates Variables along row axis.

        Args:
            variables (list): List of Variables to merge. Variables can have
                different names (and all Variables that share a name will be
                concatenated together).

        Returns:
            A list of Variables.
        '''
        var_dict = defaultdict(list)
        for v in variables:
            var_dict[v.name].append(v)
        return [merge_variables(vars_, **kwargs)
                for vars_ in list(var_dict.values())]

    def to_df(self, variables=None, format='wide', fillna=0, **kwargs):
        ''' Merge variables into a single pandas DataFrame.

        Args:
            variables (list): Optional list of column names to retain; if None,
                all variables are returned.
            format (str): Whether to return a DataFrame in 'wide' or 'long'
                format. In 'wide' format, each row is defined by a unique
                onset/duration, and each variable is in a separate column. In
                'long' format, each row is a unique combination of onset,
                duration, and variable name, and a single 'amplitude' column
                provides the value.
            fillna: Replace missing values with the specified value.
            kwargs: Optional keyword arguments to pass onto each Variable's
                to_df() call (e.g., condition, entities, and timing).

        Returns: A pandas DataFrame.
        '''

        if variables is None:
            variables = list(self.variables.keys())

        # Can receive already-selected Variables from sub-classes
        if not isinstance(variables[0], BIDSVariable):
            variables = [v for v in self.variables.values()
                         if v.name in variables]

        dfs = [v.to_df(**kwargs) for v in variables]
        df = pd.concat(dfs, axis=0)

        if format == 'long':
            return df.reset_index(drop=True).fillna(fillna)

        ind_cols = list(set(df.columns) - {'condition', 'amplitude'})
        df = df.pivot_table(index=ind_cols, columns='condition',
                            values='amplitude', aggfunc='first')
        df = df.reset_index().fillna(fillna)
        df.columns.name = None
        return df

    def clone(self):
        ''' Returns a shallow copy of the current instance, except that all
        variables are deep-cloned.
        '''
        clone = copy(self)
        clone.variables = {k: v.clone() for (k, v) in self.variables.items()}
        return clone

    def get_entities(self):
        ''' Returns a dict of entities for the current Collection.

        Note: Only entity key/value pairs common to all rows in all contained
            Variables are returned. E.g., if a Collection contains Variables
            extracted from runs 1, 2 and 3 from subject '01', the returned dict
            will be {'subject': '01'}; the runs will be excluded as they vary
            across the Collection contents.
        '''

        all_ents = pd.concat([v.entities
                              for v in self.variables.values()], axis=0)
        constant = all_ents.apply(lambda x: x.nunique() == 1)
        keep = all_ents.columns[constant]
        return {k: all_ents[k].iloc[0] for k in keep}


    # def aggregate(self, level, agg_func='mean', categorical_agg_func=None):
    #     ''' Aggregate variable values from a lower level at a higher level.

    #     Args:
    #         level (str): The level of aggregation. The returned collection will
    #             have one row per value of this level.
    #         agg_func (str, Callable): Aggregation function to use. Must be
    #             either a named function recognized by apply() in pandas, or a
    #             Callable that takes a DataFrame and returns a Series or
    #             DataFrame.
    #         categorical_agg_func (str, Callable): Aggregation function to use
    #             for categorical variables. Must be a function that returns
    #             valid output given categorical inputs. If None, aggregation
    #             will only proceed if all categorical columns have exactly one
    #             unique value.
    #     '''

    #     for var in self.variables.values():
    #         if is_numeric_dtype(var.values):
    #             _func = agg_func
    #         else:
    #             if categorical_agg_func is not None:
    #                 _func = categorical_agg_func
    #             elif var.values.nunique() > 1:
    #                 msg = ("Column %s is categorical and has more than one "
    #                        "unique value. You must explicitly specify an "
    #                        "aggregation function in the categorical_agg_func "
    #                        "argument.")
    #                 raise ValueError(msg)
    #             else:
    #                 _func = 'first'

    #         self[var.name] = var.aggregate(level, _func)

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


class BIDSRunVariableCollection(BIDSVariableCollection):

    ''' A container for one or more RunVariables--i.e., Variables that have a
    temporal dimension.

    Args:
        variables (list): A list of SparseRunVariable and/or DenseRunVariable.
        sampling_rate (float): Sampling rate (in Hz) to use when working with
            dense representations of variables. If None, defaults to 10.

    Notes:
        * Variables in the list must all be at the 'run' level. For other
          levels (session, subject, or dataset), use the
          BIDSVariableCollection.
    '''

    def __init__(self, variables, sampling_rate=None):
        # Don't put the default value in signature because None is passed from
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

    def to_df(self, variables=None, format='wide', sparse=True,
              sampling_rate=None, **kwargs):
        ''' Merge columns into a single pandas DataFrame.

        Args:
            variables (list): Optional list of variable names to retain;
                if None, all variables are written out.
            format (str): Whether to return a DataFrame in 'wide' or 'long'
                format. In 'wide' format, each row is defined by a unique
                onset/duration, and each variable is in a separate column. In
                'long' format, each row is a unique combination of onset,
                duration, and variable name, and a single 'amplitude' column
                provides the value.
            sparse (bool): If True, variables will be kept in a sparse format
                provided they are all internally represented as such. If False,
                a dense matrix (i.e., uniform sampling rate for all events)
                will be exported. Will be ignored if at least one variable is
                dense.
            sampling_rate (float): If a dense matrix is written out, the
                sampling rate (in Hz) to use for downsampling. Defaults to the
                value currently set in the instance.
            kwargs: Optional keyword arguments to pass onto each Variable's
                to_df() call (e.g., condition, entities, and timing).

        Returns: A pandas DataFrame.
        '''

        if not (sparse and self._none_dense()):
            sampling_rate = sampling_rate or self.sampling_rate

            # Make sure all variables have the same sampling rate
            variables = list(self.resample(sampling_rate, force_dense=True,
                                           in_place=False).values())

        return super(BIDSRunVariableCollection, self).to_df(variables, format,
                                                            **kwargs)


def merge_collections(collections, force_dense=False, sampling_rate='auto'):
    ''' Merge two or more collections at the same level of analysis.

    Args:
        collections (list): List of Collections to merge.
        sampling_rate (int, str): Sampling rate to use if it becomes necessary
            to resample DenseRunVariables. Either an integer or 'auto' (see
            merge_variables docstring for further explanation).

    Returns:
        A BIDSVariableCollection or BIDSRunVariableCollection, depending
        on the type of the input collections.
    '''
    if len(listify(collections)) == 1:
        return collections

    levels = set([c.level for c in collections])
    if len(levels) > 1:
        raise ValueError("At the moment, it's only possible to merge "
                         "Collections at the same level of analysis. You "
                         "passed collections at levels: %s." % levels)

    variables = list(chain(*[c.variables.values() for c in collections]))
    cls = collections[0].__class__

    variables = cls.merge_variables(variables, sampling_rate=sampling_rate)

    if isinstance(collections[0], BIDSRunVariableCollection):
        return cls(variables, sampling_rate)

    return cls(variables)
