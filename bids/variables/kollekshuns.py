''' Classes and functions related to the management of sets of BIDSVariables.

Why 'kollekshuns'? Because 'collections' would conflict with the standard lib
module of the same name on Python 2. We could go with something sensible but
verbose like 'variable_collections', but that would make far too much sense.
'''

import pandas as pd
from copy import copy
from pandas.api.types import is_numeric_dtype
import warnings
import re
from .variables import (SparseRunVariable, SimpleVariable, DenseRunVariable,
                        merge_variables, BIDSVariable)
from collections import OrderedDict
from itertools import chain
from bids.utils import listify, matches_entities
import numpy as np


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

        var_levels = set([SOURCE_TO_LEVEL[v.source] if v.source in
                          SOURCE_TO_LEVEL else v.source for v in variables])

        # TODO: relax this requirement & allow implicit merging between levels
        if len(var_levels) > 1:
            raise ValueError("A Collection cannot be initialized from "
                             "variables at more than one level of analysis. "
                             "Levels found in input variables: %s" %
                             var_levels)

        self.level = list(var_levels)[0]
        variables = self.merge_variables(variables)
        self.variables = {v.name: v for v in variables}
        self._index_entities()

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
        var_dict = OrderedDict()
        for v in variables:
            if v.name not in var_dict:
                var_dict[v.name] = []
            var_dict[v.name].append(v)
        return [merge_variables(vars_, **kwargs)
                for vars_ in list(var_dict.values())]

    def to_df(self, variables=None, format='wide', fillna=np.nan, **kwargs):
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

    @classmethod
    def from_df(cls, data, entities=None, source='contrast'):
        ''' Create a Collection from a pandas DataFrame.

        Args:
            df (DataFrame): The DataFrame to convert to a Collection. Each
                column will be converted to a SimpleVariable.
            entities (DataFrame): An optional second DataFrame containing
                entity information.
            source (str): The value to set as the source for all Variables.

        Returns:
            A BIDSVariableCollection.
        '''
        variables = []
        for col in data.columns:
            _data = pd.DataFrame(data[col].values, columns=['amplitude'])
            if entities is not None:
                _data = pd.concat([_data, entities], axis=1)
            variables.append(SimpleVariable(col, _data, source))
        return BIDSVariableCollection(variables)

    def clone(self):
        ''' Returns a shallow copy of the current instance, except that all
        variables are deep-cloned.
        '''
        clone = copy(self)
        clone.variables = {k: v.clone() for (k, v) in self.variables.items()}
        return clone

    def matches_entities(self, entities, strict=False):
        ''' Checks whether current Collection's entities match the input. '''
        return matches_entities(self, entities, strict)

    def _index_entities(self):
        ''' Sets current instance's entities based on the existing index.

        Note: Only entity key/value pairs common to all rows in all contained
            Variables are returned. E.g., if a Collection contains Variables
            extracted from runs 1, 2 and 3 from subject '01', the returned dict
            will be {'subject': '01'}; the runs will be excluded as they vary
            across the Collection contents.
        '''
        all_ents = pd.DataFrame.from_records(
            [v.entities for v in self.variables.values()])
        constant = all_ents.apply(lambda x: x.nunique() == 1)
        if constant.empty:
            self.entities = {}
        else:
            keep = all_ents.columns[constant]
            ents = {k: all_ents[k].iloc[0] for k in keep}
            self.entities = {k: v for k, v in ents.items() if pd.notnull(v)}

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

    def resample(self, sampling_rate=None, variables=None, force_dense=False,
                 in_place=False, kind='linear'):
        ''' Resample all dense variables (and optionally, sparse ones) to the
        specified sampling rate.

        Args:
            sampling_rate (int, float): Target sampling rate (in Hz). If None,
                uses the instance sampling rate.
            variables (list): Optional list of Variables to resample. If None,
                all variables are resampled.
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

        _variables = {}

        for name, var in self.variables.items():
            if variables is not None and name not in variables:
                continue
            if isinstance(var, SparseRunVariable):
                if force_dense and is_numeric_dtype(var.values):
                    _variables[name] = var.to_dense(sampling_rate)
            else:
                _variables[name] = var.resample(sampling_rate,
                                                inplace=in_place,
                                                kind=kind)
                _variables[name] = var

        if in_place:
            for k, v in _variables.items():
                self.variables[k] = v
            self.sampling_rate = sampling_rate
        else:
            return _variables

    def to_df(self, variables=None, format='wide', sparse=True,
              sampling_rate=None, include_sparse=True, include_dense=True,
              **kwargs):
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
            sparse (bool): If True, variables will be kept in a sparse
                format provided they are all internally represented as such.
                If False, a dense matrix (i.e., uniform sampling rate for all
                events) will be exported. Will be ignored if at least one
                variable is dense.
            sampling_rate (float): If a dense matrix is written out, the
                sampling rate (in Hz) to use for downsampling. Defaults to the
                value currently set in the instance.
            kwargs: Optional keyword arguments to pass onto each Variable's
                to_df() call (e.g., condition, entities, and timing).
            include_sparse (bool): Whether or not to include sparse Variables.
            include_dense (bool): Whether or not to include dense Variables.

        Returns: A pandas DataFrame.
        '''

        if not include_sparse and not include_dense:
            raise ValueError("You can't exclude both dense and sparse "
                             "variables! That leaves nothing!")

        if variables is None:
            variables = list(self.variables.keys())

        if not include_sparse:
            variables = [v for v in variables if
                         isinstance(self.variables[v], DenseRunVariable)]

        if not include_dense:
            variables = [v for v in variables if not
                         isinstance(self.variables[v], DenseRunVariable)]

        if not variables:
            return None

        _vars = [self.variables[v] for v in variables]
        if sparse and all(isinstance(v, SimpleVariable) for v in _vars):
            variables = _vars

        else:
            sampling_rate = sampling_rate or self.sampling_rate

            # Make sure all variables have the same sampling rate
            variables = list(self.resample(sampling_rate, variables,
                                           force_dense=True,
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
