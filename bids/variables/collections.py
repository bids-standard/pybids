"""Classes and functions related to the management of sets of BIDSVariables."""
from copy import copy
import warnings
import re
from collections import OrderedDict
from itertools import chain
import fnmatch

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from .variables import (SparseRunVariable, SimpleVariable, DenseRunVariable,
                        merge_variables, BIDSVariable)
from bids.utils import listify, matches_entities


class BIDSVariableCollection(object):
    """A container for one or more variables extracted from variable files
    at a single level of analysis.

    Parameters
    ----------
    variables : list
        A list of BIDSVariables or SimpleVariables.

    Notes
    -----
    Variables in the list must all share the same analysis level, which
    must be one of 'session', 'subject', or 'dataset' level. For
    run-level Variables, use the BIDSRunVariableCollection.
    """

    def __init__(self, variables):

        if not variables:
            raise ValueError("No variables were provided")
        SOURCE_TO_LEVEL = {
            'events': 'run',
            'physio': 'run',
            'stim': 'run',
            'regressors': 'run',
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
        elif not var_levels:
            raise ValueError(
                "None of the provided variables matched any of the known levels, which are: %s"
                % (', '.join(sorted(SOURCE_TO_LEVEL.values())))
            )

        self.level = list(var_levels)[0]
        variables = self.merge_variables(variables)
        self.variables = {v.name: v for v in variables}
        self._index_entities()

        # Container for variable groups (see BIDS-StatsModels spec)--maps from
        # group names to lists of variables.
        self.groups = {}

    @staticmethod
    def merge_variables(variables, **kwargs):
        """Concatenates Variables along row axis.

        Parameters
        ----------
        variables : list
            List of Variables to merge. Variables can have
            different names (and all Variables that share a name will be
            concatenated together).

        Returns
        -------
        list
            A list of Variables.
        """
        var_dict = OrderedDict()
        for v in variables:
            if v.name not in var_dict:
                var_dict[v.name] = []
            var_dict[v.name].append(v)
        return [merge_variables(vars_, **kwargs)
                for vars_ in list(var_dict.values())]

    def to_df(self, variables=None, format='wide', fillna=np.nan, **kwargs):
        """Merge variables into a single pandas DataFrame.

        Parameters
        ----------
        variables : list of str or BIDSVariable
            Optional list of variables or variable names to retain. If strings
            are passed, each one gives the name of a variable in the current
            collection. If BIDSVariables are passed, they will be used as-is.
            If None, all variables are returned. Strings and BIDSVariables
            cannot be mixed in the list.
        format : {'wide', 'long'}
            Whether to return a DataFrame in 'wide' or 'long'
            format. In 'wide' format, each row is defined by a unique
            entity combination, and each variable is in a separate column.
            In 'long' format, each row is a unique combination of entities
            and variable names, and a single 'amplitude' column provides
            the value.
        fillna : value
            Replace missing values with the specified value.
        kwargs : dict
            Optional keyword arguments to pass onto each Variable's
            to_df() call (e.g., condition, entities, and timing).

        Returns
        -------
        :obj:`pandas.DataFrame`
            A pandas DataFrame.
        """

        if variables is None:
            variables = list(self.variables.keys())

        # Can receive already-selected Variables from sub-classes
        if not isinstance(variables[0], BIDSVariable):
            variables = [v for v in self.variables.values()
                         if v.name in variables]

        dfs = [v.to_df(**kwargs) for v in variables]
        # Always concatenate along row axis, even for wide format. Then we
        # reset the index if format == 'long', which will produce the desired
        # behavior when we pivot the table.
        df = pd.concat(dfs, axis=0, sort=True)

        if format == 'long':
            return df.reset_index(drop=True).fillna(fillna)

        ind_cols = list(set(df.columns) - {'condition', 'amplitude'})

        df['amplitude'] = df['amplitude'].fillna('n/a')
        df = df.pivot_table(index=ind_cols, columns='condition',
                            values='amplitude', aggfunc='first')
        df = df.reset_index().replace('n/a', fillna)
        df.columns.name = None
        return df

    @classmethod
    def from_df(cls, data, entities=None, source='contrast'):
        """Create a Collection from a pandas DataFrame.

        Parameters
        ----------
        df : :obj:`pandas.DataFrame`
            The DataFrame to convert to a Collection. Each
            column will be converted to a SimpleVariable.
        entities : :obj:`pandas.DataFrame`
            An optional second DataFrame containing
            entity information.
        source : str
            The value to set as the source for all Variables.

        Returns
        -------
        BIDSVariableCollection
        """
        variables = []
        for col in data.columns:
            _data = pd.DataFrame(data[col].values, columns=['amplitude'])
            if entities is not None:
                _data = pd.concat([_data, entities], axis=1, sort=True)
            variables.append(SimpleVariable(name=col, data=_data, source=source))
        return BIDSVariableCollection(variables)

    def clone(self):
        """Returns a shallow copy of the current instance, except that all
        variables are deep-cloned.
        """
        clone = copy(self)
        clone.variables = {k: v.clone() for (k, v) in self.variables.items()}
        return clone


    def _index_entities(self):
        """Sets current instance's entities based on the existing index.

        Notes
        -----
        Only entity key/value pairs common to all rows in all contained
        Variables are returned. E.g., if a Collection contains Variables
        extracted from runs 1, 2 and 3 from subject '01', the returned dict
        will be {'subject': '01'}; the runs will be excluded as they vary
        across the Collection contents.
        """
        all_ents = pd.DataFrame.from_records(
            [v.entities for v in self.variables.values()])
        constant = all_ents.apply(lambda x: x.nunique() == 1)
        if constant.empty:
            self.entities = {}
        else:
            keep = all_ents.columns[constant]
            ents = {k: all_ents[k].dropna().iloc[0] for k in keep}
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

    def match_variables(self, pattern, return_type='name', match_type='unix'):
        """Return columns whose names match the provided pattern.

        Parameters
        ----------
        pattern : str, list
            One or more regex patterns to match all variable names against.
        return_type : {'name', 'variable'}
            What to return. Must be one of:
            'name': Returns a list of names of matching variables.
            'variable': Returns a list of Variable objects whose names
            match.
        match_type : str
            Matching approach to use. Either 'regex' (full-blown regular
                expression matching) or 'unix' (unix-style pattern matching
                via the fnmatch module).

        Returns
        -------
        A list of all matching variables or variable names
        """
        pattern = listify(pattern)
        results = []
        for patt in pattern:
            if match_type.lower().startswith('re'):
                patt = re.compile(patt)
                vars_ = [v for v in self.variables.keys() if patt.search(v)]
            else:
                vars_ = fnmatch.filter(list(self.variables.keys()), patt)
            if return_type.startswith('var'):
                vars_ = [self.variables[v] for v in vars_]
            results.extend(vars_)
        return results


class BIDSRunVariableCollection(BIDSVariableCollection):
    """A container for one or more RunVariables--i.e., Variables that have a
    temporal dimension.

    Parameters
    ----------
    variables : list
        A list of SparseRunVariable and/or DenseRunVariable.
    sampling_rate : float
        Sampling rate (in Hz) to use when working with
        dense representations of variables. If None, defaults to 10.

    Notes
    -----
    Variables in the list must all be at the 'run' level. For other
    levels (session, subject, or dataset), use the
    BIDSVariableCollection.
    """

    def __init__(self, variables, sampling_rate=None):
        # Don't put the default value in signature because None is passed from
        # several places and we don't want multiple conflicting defaults.
        if sampling_rate:
            if isinstance(sampling_rate, str):
                raise ValueError("Sampling rate must be numeric.")
        self.sampling_rate = sampling_rate or 10
        super(BIDSRunVariableCollection, self).__init__(variables)

    def get_dense_variables(self, variables=None):
        """Returns a list of all stored DenseRunVariables."""
        if variables is None:
            variables = set(self.variables.keys())
        return [v for v in self.variables.values()
                if isinstance(v, DenseRunVariable) and v.name in variables]

    def get_sparse_variables(self, variables=None):
        """Returns a list of all stored SparseRunVariables."""
        if variables is None:
            variables = set(self.variables.keys())
        return [v for v in self.variables.values()
                if isinstance(v, SparseRunVariable) and v.name in variables]

    def all_dense(self):
        return len(self.get_dense_variables()) == len(self.variables)

    def all_sparse(self):
        return len(self.get_sparse_variables()) == len(self.variables)

    def _get_sampling_rate(self, sampling_rate):
        """Parse sampling rate argument and return appropriate value."""

        if sampling_rate is None:
            return self.sampling_rate

        if isinstance(sampling_rate, (float, int)):
            return sampling_rate

        if sampling_rate == 'TR':
            trs = {var.run_info[0].tr for var in self.variables.values()}
            if not trs:
                raise ValueError("Repetition time unavailable; specify "
                                    "sampling_rate in Hz explicitly or set to"
                                    " 'highest'.")
            elif len(trs) > 1:
                raise ValueError("Non-unique Repetition times found "
                                    "({!r}); specify sampling_rate explicitly"
                                    .format(trs))
            return 1. / trs.pop()

        if sampling_rate.lower() == 'highest':
            dense_vars = self.get_dense_variables()
            if not dense_vars:
                return None
            return max(*[v.sampling_rate for v in dense_vars])

        raise ValueError("Invalid sampling_rate value '{}' provided. Must be "
                         "a float, None, 'TR', or 'highest'."
                         .format(sampling_rate))

    def _densify_and_resample(self, sampling_rate=None, variables=None,
                              resample_dense=False, force_dense=False,
                              in_place=False, kind='linear'):

        sampling_rate = self._get_sampling_rate(sampling_rate)

        _dense, _sparse = [], []

        # Filter variables and sort by class
        for name, var in self.variables.items():
            if variables is not None and name not in variables:
                continue
            if isinstance(var, DenseRunVariable):
                _dense.append(var)
            else:
                _sparse.append(var)

        _variables = {}

        if force_dense:
            for v in _sparse:
                if is_numeric_dtype(v.values):
                    _variables[v.name] = v.to_dense(sampling_rate)

        if resample_dense:
            for v in _dense:
                _variables[v.name] = v.resample(sampling_rate, kind=kind)

        coll = self if in_place else self.clone()

        if in_place:
            coll.variables.update(_variables)
        else:
            coll.variables = _variables

        coll.sampling_rate = sampling_rate
        return coll

    def to_dense(self, sampling_rate=None, variables=None, in_place=False,
                 kind='linear'):
        """ Convert all contained SparseRunVariables to DenseRunVariables.

        Parameters
        ----------
        sampling_rate : None, {'TR', 'highest'}, float
            Sampling rate to use when densifying sparse variables. If None,
            uses the currently stored instance value. If 'TR', the repetition
            time is used, if available, to select the sampling rate (1/TR).
            If 'highest', all variables are resampled to the highest sampling
            rate of any of the existing dense variables. The sampling rate may
            also be specified explicitly in Hz as a float.
        variables : list
            Optional list of names of Variables to resample. If None, all
            variables are resampled.
        in_place : bool
            When True, all variables are overwritten in-place.
            When False, returns resampled versions of all variables.
        kind : str
            Argument to pass to scipy's interp1d; indicates the kind of
            interpolation approach to use. See interp1d docs for valid values.

        Returns
        -------
        A BIDSVariableCollection (if in_place is False).

        Notes
        -----
        Categorical variables are ignored.
        """
        return self._densify_and_resample(sampling_rate, variables,
                                          resample_dense=False,
                                          in_place=in_place, kind=kind,
                                          force_dense=True)

    def resample(self, sampling_rate=None, variables=None, force_dense=False,
                 in_place=False, kind='linear'):
        """Resample all dense variables (and optionally, sparse ones) to the
        specified sampling rate.

        Parameters
        ----------
        sampling_rate : int or float
            Target sampling rate (in Hz). If None, uses the instance value.
        variables : list
            Optional list of names of Variables to resample. If None, all
            variables are resampled.
        force_dense : bool
            if True, all sparse variables will be forced to dense.
        in_place : bool
            When True, all variables are overwritten in-place.
            When False, returns resampled versions of all variables.
        kind : str
            Argument to pass to scipy's interp1d; indicates the kind of
            interpolation approach to use. See interp1d docs for valid values.

        Returns
        -------
        A BIDSVariableCollection (if in_place is False).
        """
        return self._densify_and_resample(sampling_rate, variables,
                                   force_dense=force_dense, in_place=in_place,
                                   kind=kind, resample_dense=True)

    def to_df(self, variables=None, format='wide', sparse=True,
              sampling_rate=None, include_sparse=True, include_dense=True,
              **kwargs):
        """Merge columns into a single pandas DataFrame.

        Parameters
        ----------
        variables : list
            Optional list of variable names to retain;
            if None, all variables are written out.
        format : str
            Whether to return a DataFrame in 'wide' or 'long'
            format. In 'wide' format, each row is defined by a unique
            onset/duration, and each variable is in a separate column. In
            'long' format, each row is a unique combination of onset,
            duration, and variable name, and a single 'amplitude' column
            provides the value.
        sparse : bool
            If True, variables will be kept in a sparse
            format provided they are all internally represented as such.
            If False, a dense matrix (i.e., uniform sampling rate for all
            events) will be exported. Will be ignored if at least one
            variable is dense.
        sampling_rate : float
            If a dense matrix is written out, the
            sampling rate (in Hz) to use for downsampling. Defaults to the
            value currently set in the instance.
        kwargs : dict
            Optional keyword arguments to pass onto each Variable's
            to_df() call (e.g., condition, entities, and timing).
        include_sparse : bool
            Whether or not to include sparse Variables.
        include_dense : bool
            Whether or not to include dense Variables.

        Returns
        -------
        :obj:`pandas.DataFrame`
            A pandas DataFrame.
        """

        if not include_sparse and not include_dense:
            raise ValueError("You can't exclude both dense and sparse "
                             "variables! That leaves nothing!")

        _vars = []

        if include_sparse:
            _vars += self.get_sparse_variables(variables)

        if include_dense:
            _vars += self.get_dense_variables(variables)

        if not _vars:
            return None

        # If all variables are sparse/simple, we can pass them as-is. Otherwise
        # we first force all variables to dense via .resample().
        if sparse and all(isinstance(v, SimpleVariable) for v in _vars):
            variables = _vars
        else:
            sampling_rate = sampling_rate or self.sampling_rate
            var_names = [v.name for v in _vars]
            collection = self.resample(sampling_rate, variables=var_names,
                                       force_dense=True)
            variables = list(collection.variables.values())

        return super(BIDSRunVariableCollection, self).to_df(variables, format,
                                                            **kwargs)


def merge_collections(collections, sampling_rate='highest'):
    """Merge two or more collections at the same level of analysis.

    Parameters
    ----------
    collections : list
        List of Collections to merge.
    sampling_rate : int or str
        Sampling rate to use if it becomes necessary
        to resample DenseRunVariables. Either an integer or 'highest' (see
        merge_variables docstring for further explanation).

    Returns
    -------
    BIDSVariableCollection or BIDSRunVariableCollection
        Result type depends on the type of the input collections.
    """
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
        # 'auto' was renamed to 'highest' circa 0.10, but check for both
        if sampling_rate in {'auto', 'highest'}:
            rates = [var.sampling_rate for var in variables
                     if isinstance(var, DenseRunVariable)]

            sampling_rate = rates[0] if rates else None

        return cls(variables, sampling_rate)

    return cls(variables)
