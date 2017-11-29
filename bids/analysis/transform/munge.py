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
        do is unset the original column in the manager. '''
        self.manager.columns.pop(col.name)
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
    _columns_used = ('cols', 'by')
    _densify = ('cols', 'by')
    _allow_categorical = ('by',)

    def _transform(self, col, by):
        from bids.analysis.variables import SparseBIDSColumn

        if not isinstance(col, SparseBIDSColumn):
            self._densify_columns()

        # Set up all the splitting columns as a DF
        group_data = pd.concat([self._columns[c].values for c in listify(by)],
                               axis=1)
        group_data.columns = listify(by)

        # For sparse data, we need to set up a 1D grouper
        if isinstance(col, SparseBIDSColumn):
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

        if not input_attr.endswith('s'):
            input_attr += 's'

        if not target_attr.endswith('s'):
            target_attr += 's'

        # assign only makes sense for sparse columns; dense columns don't have
        # durations or onsets, and amplitudes can be copied by cloning
        from bids.analysis.variables import DenseBIDSColumn
        if isinstance(input, DenseBIDSColumn):
            raise ValueError("The 'assign' transformation can only be applied"
                             " to sparsely-coded event types. The input "
                             "column (%s) is dense." % input.name)

        target = self.manager.columns[target].clone()
        if isinstance(target, DenseBIDSColumn):
            raise ValueError("The 'assign' transformation can only be applied"
                             " to sparsely-coded event types. The target "
                             "column (%s) is dense." % target.name)

        # Ensure attributes are valid
        valid_attrs = ['amplitudes', 'durations', 'onsets']
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

    def _transform(self, col, constraint='none', ref_level=None):

        from bids.analysis.variables import SparseBIDSColumn

        result = []
        data = col.to_df()
        grps = data.groupby('amplitude')
        orig_name = col.name

        # Determine the reference level
        if constraint in ['drop_one', 'mean_zero']:
            levels = data['amplitude'].unique().sort_values()
            if ref_level is None:
                ref_level = levels[0]

        for i, (lev_name, lev_grp) in enumerate(grps):
            # TODO: consider appending info about the constraint to the name,
            # though this has the downside of making names very long and
            # difficult to work with.
            name = '%s/%s' % (col.name, lev_name)
            # TODO: implement constraint == 'mean_zero'
            if constraint == 'drop_one' and lev_name == ref_level:
                continue
            lev_grp['amplitude'] = 1.0
            col = SparseBIDSColumn(self.manager, name, lev_grp,
                                   factor_name=col.name, factor_index=i,
                                   level_name=lev_name)
            result.append(col)

        # Remove existing column. TODO: allow user to leave original in?
        self.manager.columns.pop(orig_name)

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

        data = pd.concat([self.manager[c].values for c in names], axis=1)
        data = data.reset_index(drop=True) # Make sure we can use integer index
        data.columns = list(name_map.values())
        data = data.query(query)

        # Truncate target column to retained rows
        col.onsets = col.onsets[data.index]
        col.durations = col.durations[data.index]
        col.values = col.values.iloc[data.index]

        return col


class select(Transformation):
    pass


class replace(Transformation):
    pass
