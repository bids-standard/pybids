'''
Transformations that primarily involve manipulating/munging variables into
other formats or shapes.
'''

import numpy as np
import pandas as pd
from bids.utils import listify
from .base import Transformation
from patsy import dmatrix
import re
from bids.variables import DenseRunVariable, SimpleVariable


class copy(Transformation):
    ''' Copy/clone a variable.

    Args:
        col (str): Name of variable to copy.
    '''

    _groupable = False
    _output_required = True
    _allow_categorical = ('variables',)

    def _transform(self, col):
        # We don't have to do anything else here b/c it's handled in base.
        return col


class rename(Transformation):
    ''' Rename a variable.

    Args:
        var (str): Name of existing variable to rename.
    '''
    _groupable = False
    _output_required = True
    _input_type = 'variable'
    _allow_categorical = ('variables',)

    def _transform(self, var):
        ''' Rename happens automatically in the base class, so all we need to
        do is unset the original variable in the collection. '''
        self.collection.variables.pop(var.name)
        return var.values.values


class split(Transformation):
    ''' Split a single variable into N variables as defined by the levels of one or
    more other variables.

    Args:
        by (str, list): Name(s) of variable(s) to split on.
    '''

    _variables_used = ('variables', 'by')
    _groupable = False
    _input_type = 'variable'
    _return_type = 'variable'
    _allow_categorical = ('by',)
    _densify = ('variables', 'by')

    def _transform(self, var, by, drop_orig=True):

        if not isinstance(var, SimpleVariable):
            self._densify_variables()

        # Set up all the splitting variables as a DF. Note that variables in
        # 'by' can be either regular variables, or entities in the index--so
        # we need to check both places.
        all_variables = self._variables
        by_variables = [all_variables[v].values if v in all_variables
                        else var.index[v].reset_index(drop=True)
                        for v in listify(by)]
        group_data = pd.concat(by_variables, axis=1)
        group_data.columns = listify(by)

        # For sparse data, we need to set up a 1D grouper
        if isinstance(var, SimpleVariable):
            # Create single grouping variable by combining all 'by' variables
            if group_data.shape[1] == 1:
                group_labels = group_data.iloc[:, 0].values
            else:
                group_rows = group_data.astype(str).values.tolist()
                group_labels = ['_'.join(r) for r in group_rows]

            result = var.split(group_labels)

        # For dense data, use patsy to create design matrix, then multiply
        # it by target variable
        else:
            group_data = group_data.astype(str)
            formula = '0+' + '*'.join(listify(by))
            dm = dmatrix(formula, data=group_data, return_type='dataframe')
            result = var.split(dm)

        if drop_orig:
            self.collection.variables.pop(var.name)

        return result


class to_dense(Transformation):
    ''' Convert variable to dense representation. '''

    _groupable = False
    _input_type = 'variable'
    _return_type = 'variable'

    def _transform(self, var):
        return var.to_dense()


class assign(Transformation):
    ''' Assign one variable's amplitude, duration, or onset attribute to
    another. '''

    _groupable = False
    _input_type = 'variable'
    _return_type = 'variable'
    _allow_categorical = ('variables', 'target')

    def _transform(self, input, target, input_attr='amplitude',
                   target_attr='amplitude'):

        # assign only makes sense for sparse variables; dense variables don't
        # have durations or onsets, and amplitudes can be copied by cloning
        if isinstance(input, DenseRunVariable):
            raise ValueError("The 'assign' transformation can only be applied"
                             " to sparsely-coded event types. The input "
                             "variable (%s) is dense." % input.name)

        target = self.collection.variables[target].clone()
        if isinstance(target, DenseRunVariable):
            raise ValueError("The 'assign' transformation can only be applied"
                             " to sparsely-coded event types. The target "
                             "variable (%s) is dense." % target.name)

        # Ensure attributes are valid
        valid_attrs = ['amplitude', 'duration', 'onset']
        if input_attr not in valid_attrs:
            raise ValueError("Valid values for input_attr are: %s." %
                             valid_attrs)
        if target_attr not in valid_attrs:
            raise ValueError("Valid values for target_attr are: %s." %
                             valid_attrs)

        # variables must have same number of events, but do *not* need to have
        # aligned onsets.
        l_s, l_t = len(input.values), len(target.values)
        if l_s != l_t:
            raise ValueError("Input and target variables do not contain the "
                             "same number of events (%d vs. %d)." % (l_s, l_t))

        if input_attr.startswith('amplitude'):
            vals = input.values.values
        else:
            vals = getattr(input, input_attr)

        if target_attr.startswith('amplitude'):
            target.values[:] = vals
        else:
            setattr(target, target_attr, vals)

        return target


class factor(Transformation):

    _groupable = False
    _input_type = 'variable'
    _return_type = 'variable'
    _allow_categorical = ('variables',)

    def _transform(self, var, constraint='none', ref_level=None, sep='.'):

        result = []
        data = var.to_df()
        orig_name = var.name
        variableClass = var.__class__

        levels = np.sort(data['amplitude'].unique())
        new_cols = pd.get_dummies(data['amplitude'], drop_first=False)[levels]

        if len(levels) > 1 and constraint in ('drop_one', 'mean_zero'):
            if ref_level is None:
                ref_level = levels[0]
            new_cols = new_cols.drop(ref_level, axis=1)

            if constraint == 'mean_zero':
                ref_inds = data['amplitude'] == ref_level
                new_cols.loc[ref_inds, :] = -1. / (len(levels) - 1)

        for lev in levels:
            if ref_level is not None and lev == ref_level:
                continue
            name = ''.join([var.name, sep, str(lev)])
            lev_data = data.copy()
            lev_data['amplitude'] = new_cols[lev].astype(float)
            args = [name, lev_data, var.source]
            if hasattr(var, 'run_info'):
                args.insert(2, var.run_info)
            new_col = variableClass(*args)
            result.append(new_col)

        # Remove existing variable. TODO: allow user to leave original in?
        self.collection.variables.pop(orig_name)

        return result


class filter(Transformation):

    _groupable = False
    _input_type = 'variable'
    _return_type = 'variable'
    _align = ('by')
    _allow_categorical = ('variables', 'by')

    def _transform(self, var, query, by=None):

        if by is None:
            by = []

        names = [var.name] + listify(by)

        # pandas .query can't handle non-identifiers in variable names, so we
        # need to replace them in both the variable names and the query string.
        name_map = {n: re.sub('[^a-zA-Z0-9_]+', '_', n) for n in names}
        for k, v in name_map.items():
            query = query.replace(k, v)

        data = pd.concat([self.collection[c].values for c in names], axis=1)
        # Make sure we can use integer index
        data = data.reset_index(drop=True)
        data.columns = list(name_map.values())
        data = data.query(query)

        # Truncate target variable to retained rows
        var.onset = var.onset[data.index]
        var.duration = var.duration[data.index]
        var.values = var.values.iloc[data.index]

        return var


class select(Transformation):
    ''' Select variables to retain.

    Args:
        variables (list, str): Name(s) of variables to retain. All variables
            not in the list will be dropped from the collection.
    '''
    _groupable = False
    _loopable = False
    _input_type = 'variable'
    _return_type = 'none'

    def _transform(self, variables):
        self.collection.variables = {c.name: c for c in variables}


class replace(Transformation):
    pass
