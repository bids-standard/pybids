''' Base Transformation class and associated utilities. '''

import numpy as np
import pandas as pd
from bids.events.utils import listify
import warnings
from abc import ABCMeta, abstractmethod
from copy import deepcopy
import itertools
import re


class Transformation(object):

    ### Class-level settings ###
    # The following settings govern the way Transformations are applied to the
    # data. The default settings can be overridden within subclasses.

    # List all argument names that specify columns used in the Transformation.
    # This is necessary in order to ensure that all and only columns touched
    # by the transformation are cloned before any manipulation occurs.
    # Columns in 'cols' are always cloned, so only additional arguments should
    # be specified here.
    _columns_used = ()

    # What data type to pass onto the core _transform() logic. Must be one
    # of 'column' (the entire BIDSColumn object), 'pandas' (the extracted
    # pandas DF stored in .values), or 'numpy' (just the numpy array inside
    # the .values property of the pandas DF). To minimize overhead and
    # simplify code, it is recommended to avoid using 'column' if possible.
    _input_type = 'pandas'

    # The data type the internal _transform() method is expected to return.
    # Must be one of 'column', 'pandas', 'numpy', or 'none'. In the last
    # case, all desired changes must be made in-place within _transform(), as
    # no further changes will be committed.
    _return_type = 'pandas'

    # A tuple indicating which arguments give the names of columns that must
    # all be aligned with one another (i.e., onsets and durations match
    # perfectly) before processing. Defaults to None.
    _align = None

    # Boolean indicating whether the Transformation should be applied to each
    # column in the input list in turn. When True (default), the Transformation
    # is applied once per element in the column list, with all arguments
    # being passed repeatedly. When False, all data (i.e., columns or their
    # pandas DFs or ndarrays, as specified in _input_type) are passed to the
    # Transformation simultaneously.
    _loopable = True

    # Boolean indicating whether the Transformation can handle groupby
    # operations. When True, a 'groupby' argument is made implicitly available,
    # and if passed, the Transformation will be applied separately to each
    # subset of the data, as defined by the columns named in groupby. When
    # False, the Transformations does not allow grouping, and will raise an
    # exception if groupby is passed. Transformations should set this to False
    # if the groupby argument cannot possibly change the returned result.
    _groupable = True

    # Boolean indicating whether the output argument is mandatory. When False
    # (default), transformations will be applied in-place unless output is set.
    # When True, the user must explicitly specify the output, or an exception
    # is raised.
    _output_required = False

    # An implicit 'dense' argument is always available, and indicates whether
    # or not to operate on dense columns. When True, the arguments listed in
    # _densify control which columns will be densified. Defaults to the columns
    # named in the 'cols' argument. Note that if this value is overridden,
    # 'cols' will need to be explicitly included (i.e., the subclass's
    # _densify tuple replaces the base class rather than appending to it).
    _densify = ('cols',)

    __metaclass__ = ABCMeta

    def __new__(cls, collection, cols, *args, **kwargs):
        t = super(Transformation, cls).__new__(cls)
        t._setup(collection, cols, *args, **kwargs)
        return t.transform()

    def _setup(self, collection, cols, *args, **kwargs):
        ''' Replaces __init__ to set instance attributes because on Python
        >= 3.3, we can't override both new and init. '''
        self.collection = collection
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
        self._columns = {c: self.collection[c].clone() for c in self.cols}

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
                self._columns[c] = col.to_dense()

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

        # Align columns if needed
        self._align_columns(columns)

        # Pass desired type--column, DataFrame, or NDArray
        def select_type(col):
            return {'column': col, 'pandas': col.values,
                    'numpy': col.values.values}[self._input_type]

        data = [select_type(c) for c in columns]

        if not self._loopable:
            columns = [columns]

        for i, col in enumerate(columns):

            # If we still have a list, pass all columns in one block
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
            elif self._return_type == 'column':
                col = result

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

        if self._align is None or self._align == 'none':
            return

        def _align(cols):
            # If any column is dense, all columns must be dense
            from bids.events.base import SparseBIDSColumn
            sparse = [c for c in cols if isinstance(c, SparseBIDSColumn)]
            if len(sparse) < len(cols):
                if sparse:
                    sparse_names = [s.name for s in sparse]
                    msg = ("Found a mix of dense and sparse columns. This may "
                           "cause problems for some transformations.")
                    if force:
                        msg += (" Sparse columns %s were converted to dense "
                                "form to ensure proper alignment." %
                                sparse_names)
                        sparse = [s.to_dense() for s in sparse]
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
                        cols = [c.to_dense() for c in cols]
                    warnings.warn(msg)

        align_cols = [listify(self.kwargs[v]) for v in listify(self._align)
                      if v in self.kwargs]
        align_cols = list(itertools.chain(*align_cols))
        align_cols = [self.collection[c] for c in align_cols if c]

        if align_cols and self._loopable:
            for c in cols:
                # TODO: should clone all variables in align_cols before
                # alignment to prevent conversion to dense in any given
                # iteration having side effects. This could be an issue if,
                # e.g., some columns in 'cols' are dense and some are sparse.
                _align([c] + align_cols)
        else:
            _align(listify(cols) + align_cols)
