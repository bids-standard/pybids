''' Base Transformation class and associated utilities. '''

import numpy as np
import pandas as pd
from bids.events.utils import listify
import warnings
from abc import ABCMeta, abstractmethod
from copy import deepcopy
import itertools


class Transformation(object):

    _columns_used = ()          # List all columns the transformation touches
    _input_type = 'pandas'      # 'column', 'pandas', or 'numpy'
    _return_type = 'pandas'     # 'column', 'pandas', 'numpy', 'none'

    _loopable = True            # Loop over input columns one at a time?
    _groupable = True           # Can groupby operations be applied?
    _output_required = False    # Require names of output columns?
    _densify = ('cols',)           # Columns to densify, if dense=True in args

    __metaclass__ = ABCMeta

    def __new__(cls, transformer, cols, *args, **kwargs):
        t = super(Transformation, cls).__new__(cls)
        t._setup(transformer, cols, *args, **kwargs)
        return t.transform()

    def _setup(self, transformer, cols, *args, **kwargs):
        ''' Replaces __init__ to set instance attributes because on Python
        >= 3.3, we can't override both new and init. '''
        self.transformer = transformer
        self.collection = self.transformer.collection
        self.cols = listify(cols)
        self.groupby = kwargs.pop('groupby', 'event_file_id')
        self.output = listify(kwargs.pop('output', None))
        self.dense = kwargs.pop('dense', False)
        self.args = args
        self.kwargs = kwargs

    def _clone_columns(self):
        ''' Deep copy all columns the transformation touches. This prevents us
        from unnecessarily overwriting existing columns. '''

        # Always clone the target columns
        self._columns = {c: deepcopy(self.collection[c]) for c in self.cols}

        if not self._columns_used:
            return

        # Loop over argument names and clone all column names in each one
        for var in self._columns_used:
            for c in listify(self.kwargs.get(var, [])):
                self._columns[c] = deepcopy(self.collection[c])

    def _densify_columns(self):

        from bids.events.base import SparseBIDSColumn

        cols = []

        for var in self._densify:

            if var == 'cols':
                cols.extend(self.cols)
            else:
                cols.extend(listify(self.kwargs.get(var, [])))

        for c in cols:
            col = self._columns[c]
            if isinstance(col, SparseBIDSColumn):
                self._columns[c] = col.to_dense(self.transformer)

    def transform(self):

        if self.output is None and (self._output_required or not
                                    self._loopable):
            raise ValueError("Transformation '%s' requires the 'output' "
                             "argument to be set." % self.__class__.__name__)

        args, kwargs = self.args, self.kwargs

        # Deep copy all columns we expect to touch
        self._clone_columns()

        # Densify columns if needed
        if self.dense:
            self._densify_columns()

        # Set columns we plan to operate on directly
        columns = [self._columns[c] for c in self.cols]

        # Pass desired type--column, DataFrame, or NDArray
        def select_type(col):
            return {'column': col, 'pandas': col.values,
                    'numpy': col.values.values}[self._input_type]

        data = [select_type(c) for c in columns]

        if not self._loopable:
            columns = [columns]

        for i, col in enumerate(columns):

            # If we still have a list, pass all columns to the transformer
            # in one block
            if isinstance(col, (list, tuple)):
                result = self._transform(data, *args, **kwargs)
                col = col[0].clone(data=result, name=self.output[0])
            # Otherwise loop over columns individually
            else:
                if self._groupable and self.groupby is not None:
                    result = col.apply(self._transform, groupby=self.groupby,
                                       *args, **kwargs)
                else:
                    result = self._transform(data[i], *args, **kwargs)

            if self._return_type == 'none' or self._return_type is None:
                continue
            elif self._return_type == 'numpy':
                col.values = pd.DataFrame(result)
            elif self._return_type == 'pandas':
                col.values = result

            # Overwrite existing column
            if self.output is None:
                # If multiple Columns were returned, add each one separately
                if isinstance(result, (list, tuple)):
                    for r in result:
                        self.collection[r.name] = r
                else:
                    self.collection[col.name] = col

            # Set as a new column
            else:
                if len(self.cols) == len(self.output) or not self._loopable:
                    _output = self.output[i]
                elif len(self.output) == 1:
                    _output = str(self.output) + '_' + col.name
                col.name = _output
                self.collection[_output] = col

    @abstractmethod
    def _transform(self, *args, **kwargs):
        pass

    def _preprocess(self, col):
        return col

    def _postprocess(self, col):
        return col

    def _align_columns(self, cols, force=True):
        ''' Checks whether the specified columns have aligned indexes. This
        implies either that all columns are dense, or that all columns are
        sparse and have exactly the same onsets and durations. If columns are
        not aligned and force = True, all columns will be forced to dense
        format in order to ensure alignment.
        '''

        if self._align is None:  # We shouldn't be here!
            return

        def _align(cols):
            # If any column is dense, all columns must be dense
            from bids.events.base import SparseBIDSColumn
            sparse = [isinstance(c, SparseBIDSColumn) for c in cols]
            if len(sparse) < len(cols):
                if sparse:
                    sparse_names = [s.name for s in sparse]
                    msg = ("Found a mix of dense and sparse columns. This may "
                           "cause problems for some transformations.")
                    if force:
                        msg += (" Sparse columns  %s were converted to dense "
                                "form to ensure proper alignment." %
                                sparse_names)
                        sparse = [s.to_dense(self) for s in sparse]
                warnings.warn(msg)
            # If all are sparse, durations, onsets, and index must match
            # perfectly for all
            else:
                def get_col_data(col):
                    return np.c_[col.values.index, col.durations, col.onsets]

                def compare_cols(a, b):
                    return len(a) == len(b) and np.allclose(a, b)

                # Compare 1st col with each of the others
                fc = get_col_data(cols[0])
                if not all([compare_cols(fc, get_col_data(c))
                           for c in cols[1:]]):
                    msg = "Misaligned sparse columns found."
                    if force:
                        msg += (" Forcing all sparse columns to dense in order"
                                " to ensure proper alignment.")
                        cols = [c.to_dense(self) for c in cols]
                    warnings.warn(msg)

        align_cols = [self.kwargs.get(v, []) for v in listify(self._align)]
        align_cols = list(itertools.chain(align_cols))
        align_cols = [self.collection[c] for c in align_cols]

        if self._loopable:
            for c in cols:
                _align([c] + align_cols)
        else:
            _align(listify(cols) + align_cols)
