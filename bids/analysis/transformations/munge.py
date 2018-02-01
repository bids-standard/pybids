'''
Transformations that primarily involve manipulating/munging columns into other
formats or shapes.
'''

import pandas as pd
from bids.utils import listify
from .base import Transformation
from patsy import dmatrix
import re


class copy(Transformation):
    ''' Copy/clone a column.

    Args:
        col (str): Name of column to copy.
    '''

    _groupable = False
    _output_required = True
    _allow_categorical = ('cols',)

    def _transform(self, col):
        # We don't have to do anything else here b/c it's handled in base.
        return col


class rename(Transformation):
    ''' Rename a column.

    Args:
        col (str): Name of existing column to rename.
    '''
    _groupable = False
    _output_required = True
    _input_type = 'column'
    _allow_categorical = ('cols',)

    def _transform(self, col):
        ''' Rename happens automatically in the base class, so all we need to
        do is unset the original column in the collection. '''
        self.collection.columns.pop(col.name)
        return col.values.values


class split(Transformation):
    ''' Split a single column into N columns as defined by the levels of one or
    more other columns.

    Args:
        by (str, list): Name(s) of variable(s) to split on.
    '''

    _groupable = False
    _input_type = 'column'
    _return_type = 'column'
    _allow_categorical = ('by',)

    def _transform(self, col, by):
        from bids.analysis.variables import SimpleVariable

        if not isinstance(col, SimpleVariable):
            self._densify_columns()

        # Set up all the splitting columns as a DF. Note that columns in 'by'
        # can be either regular columns, or entities in the index--so we need
        # to check both places.
        all_cols = self.collection.columns
        by_cols = [all_cols[c].values if c in all_cols
                   else col.index[c].reset_index(drop=True)
                   for c in listify(by)]
        group_data = pd.concat(by_cols, axis=1)
        group_data.columns = listify(by)

        # For sparse data, we need to set up a 1D grouper
        if isinstance(col, SimpleVariable):
            # Create single grouping column by combining all 'by' columns
            if group_data.shape[1] == 1:
                group_labels = group_data.iloc[:, 0].values
            else:
                group_rows = group_data.astype(str).values.tolist()
                group_labels = ['_'.join(r) for r in group_rows]

            return col.split(group_labels)

        # For dense data, use patsy to create design matrix, then multiply
        # it by target column
        else:
            group_data = group_data.astype(str)
            formula = '0+' + '*'.join(listify(by))
            dm = dmatrix(formula, data=group_data, return_type='dataframe')
            return col.split(dm)


class to_dense(Transformation):
    ''' Convert column to dense representation. '''

    _groupable = False
    _input_type = 'column'
    _return_type = 'column'

    def _transform(self, col):
        return col.to_dense()


class assign(Transformation):
    ''' Assign one column's amplitude, duration, or onset attribute to
    another. '''

    _groupable = False
    _input_type = 'column'
    _return_type = 'column'
    _allow_categorical = ('cols', 'target')

    def _transform(self, input, target, input_attr='amplitude',
                   target_attr='amplitude'):

        # assign only makes sense for sparse columns; dense columns don't have
        # durations or onsets, and amplitudes can be copied by cloning
        from bids.analysis.variables import DenseEventVariable
        if isinstance(input, DenseEventVariable):
            raise ValueError("The 'assign' transformation can only be applied"
                             " to sparsely-coded event types. The input "
                             "column (%s) is dense." % input.name)

        target = self.collection.columns[target].clone()
        if isinstance(target, DenseEventVariable):
            raise ValueError("The 'assign' transformation can only be applied"
                             " to sparsely-coded event types. The target "
                             "column (%s) is dense." % target.name)

        # Ensure attributes are valid
        valid_attrs = ['amplitude', 'duration', 'onset']
        if input_attr not in valid_attrs:
            raise ValueError("Valid values for input_attr are: %s." %
                             valid_attrs)
        if target_attr not in valid_attrs:
            raise ValueError("Valid values for target_attr are: %s." %
                             valid_attrs)

        # Columns must have same number of events, but do *not* need to have
        # aligned onsets.
        l_s, l_t = len(input.values), len(target.values)
        if l_s != l_t:
            raise ValueError("Input and target columns do not contain the "
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
    _input_type = 'column'
    _return_type = 'column'
    _allow_categorical = ('cols',)

    def _transform(self, col, constraint='none', ref_level=None, sep='.'):

        result = []
        data = col.to_df()
        grps = data.groupby('amplitude')
        orig_name = col.name
        ColumnClass = col.__class__

        # Determine the reference level
        if constraint in ['drop_one', 'mean_zero']:
            levels = data['amplitude'].unique().sort_values()
            if ref_level is None:
                ref_level = levels[0]

        for i, (lev_name, lev_grp) in enumerate(grps):
            # TODO: consider appending info about the constraint to the name,
            # though this has the downside of making names very long and
            # difficult to work with.
            name = ''.join([col.name, sep, lev_name])
            # TODO: implement constraint == 'mean_zero'
            if constraint == 'drop_one' and lev_name == ref_level:
                continue
            lev_grp['amplitude'] = 1.0

            new_col = ColumnClass(self.collection, name, lev_grp,
                                  factor_name=col.name, level_index=i,
                                  level_name=lev_name)
            result.append(new_col)

        # Remove existing column. TODO: allow user to leave original in?
        self.collection.columns.pop(orig_name)

        return result


class filter(Transformation):

    _groupable = False
    _input_type = 'column'
    _return_type = 'column'
    _align = ('by')
    _allow_categorical = ('cols', 'by')

    def _transform(self, col, query, by=None):

        if by is None:
            by = []

        names = [col.name] + listify(by)

        # pandas .query can't handle non-identifiers in column names, so we
        # need to replace them in both the column names and the query string.
        name_map = {n: re.sub('[^a-zA-Z0-9_]+', '_', n) for n in names}
        for k, v in name_map.items():
            query = query.replace(k, v)

        data = pd.concat([self.collection[c].values for c in names], axis=1)
        # Make sure we can use integer index
        data = data.reset_index(drop=True)
        data.columns = list(name_map.values())
        data = data.query(query)

        # Truncate target column to retained rows
        col.onset = col.onset[data.index]
        col.duration = col.duration[data.index]
        col.values = col.values.iloc[data.index]

        return col


class select(Transformation):
    ''' Select columns to retain.

    Args:
        cols (list, str): Name(s) of columns to retain. All columns not in the
            list will be dropped from the collection.
    '''
    _groupable = False
    _loopable = False
    _input_type = 'column'
    _return_type = 'none'

    def _transform(self, cols):
        self.collection.columns = {c.name: c for c in cols}


class replace(Transformation):
    pass
