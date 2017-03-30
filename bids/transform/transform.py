import numpy as np
import pandas as pd
from .utils import listify
from .base import SparseBIDSColumn
import warnings
import os
import json
import math
from six import string_types
from abc import ABCMeta, abstractmethod
from copy import deepcopy
import itertools


class Transformation(object):

    _columns_used = ()          # List all columns the transformation touches
    _loopable = True            # Loop over input columns one at a time?
    _groupable = True           # Can groupby operations be applied?
    _input = 'pandas'           # Either 'pandas' or 'column'
    _output_required = False    # Require names of output columns?
    _densify = ('cols')           # Columns to densify, if dense=True in args

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

        for var in self._densify:

            if var == 'cols':
                for c in self.cols:
                    self._columns[c] = self._columns[c].to_dense(self.transformer)
            else:
                for c in listify(self.kwargs.get(var, [])):
                    self._columns[c] = self._columns[c].to_dense(self.transformer)

    def transform(self):

        if self.output is None and (self._output_required or not self._loopable):
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

        if not self._loopable:
            columns = [columns]

        for i, col in enumerate(columns):

            # Handle all columns together
            if isinstance(col, (list, tuple)):
                result = self._transform(col.values, *args, **kwargs)
                col = col.clone(data=result, name=self.output[0])
            # Loop over columns individually
            else:
                self.column = col
                if self._groupable and self.groupby is not None:
                    result = col.apply(self._transform, groupby=self.groupby,
                                       *args, **kwargs)
                else:
                    data = {
                        'column': col,
                        'pandas': col.values,
                        'numpy': col.values.values
                    }[self._input]
                    result = self._transform(data, *args, **kwargs)

            col.values = result
            # Overwrite existing column
            if self.output is None:
                self.collection[col.name] = col
            # Set as a new column
            else:
                if len(self.cols) == len(self.output):
                    _output = self.output[i]
                elif len(self.output) == 1:
                    _output = str(self.output) + '_' + col.name
                col.name = _output
                self.collection[_output] = col

    @abstractmethod
    def _transform(self, *args, **kwargs):
        pass

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
                if not all([compare_cols(fc, get_col_data(c)) for c in cols[1:]]):
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


class scale(Transformation):

    def _transform(self, data, demean=True, rescale=True):
        if demean:
            data -= data.mean(0)
        if rescale:
            data /= data.std(0)
        return data


class sum(Transformation):

    _loopable = False
    _groupable = False
    _align = True

    def _transform(self, data):
        return data.sum(0)


class orthogonalize(Transformation):

    _columns_used = ('cols', 'other')
    _densify = ('cols', 'other')
    _align = ('other')

    def _transform(self, col, other):

        other = listify(other)

        # Set up X matrix and slice into it based on target column indices
        # TODO: verify that pandas and numpy indices always align! (I.e.,
        # the pandas index should always be range(n_rows), in that order.)
        X = np.array([self._columns[c].values.values.squeeze() for c in other]).T
        X = X[col.index, :]
        assert len(X) == len(col)
        y = col.values
        _aX = np.c_[np.ones(len(y)), X]
        coefs, resids, rank, s = np.linalg.lstsq(_aX, y)
        result = pd.DataFrame(y - X.dot(coefs[1:]), index=col.index)
        return result


class rename(Transformation):
    ''' Rename a column.

    Args:
        col (str): Name of existing column to rename.
    '''
    _groupable = False
    _output_required = True
    _input = 'column'

    def _transform(self, col):
        ''' Rename happens automatically in the base class, so all we need to
        do is unset the original column in the collection. '''
        self.collection.columns.pop(col.name)
        return col.values.values


class binarize(Transformation):
    ''' Binarize a column.
    Args:
        col (Series/DF): The pandas structure to binarize.
        threshold (float): The value to binarize around (values above will
            be assigned 1, values below will be assigned 0).
    '''

    _groupable = False
    _input = 'pandas'

    def _transform(self, data, threshold=0.):
        above = data > threshold
        data[above] = 1
        data[~above] = 0
        return data


# class split(Transformation):
#     ''' Split a single column into N columns as defined by the levels of
#     one or more other columns.

#     Args:
#         col (BIDSColumn): The column whose events should be split.
#         by (str, list): The names of the columns whose levels define the groups
#             to split on.
#     '''

#     _groupable = False
#     _input = 'pandas'
#     _align = 'by'

#     def _transform(self, col, by):
#         pass


class BIDSTransformer(object):

    ''' Applies supported transformations to the columns of event files
    specified in a BIDS project.

    Args:
        collection (BIDSEventCollection): The collection to operate on.
        spec (str): An optional path to a json file containing transformation
            specifications.
        sampling_rate (int): The sampling rate (in hz) to use when working with
            dense columns. This should be sufficiently high to minimize
            information loss when resampling, but larger values will slow down
            transformation.
    '''

    def __init__(self, collection, spec=None, sampling_rate=10):
        self.collection = collection
        self.sampling_rate = sampling_rate
        self._build_dense_index()
        if spec is not None:
            self.apply_from_json(spec)

    def _build_dense_index(self):
        ''' Build an index of all tracked entities for all dense columns. '''
        index = []
        for evf in self.collection.event_files:
            reps = int(math.ceil(evf.duration * self.sampling_rate))
            ent_vals = list(evf.entities.values())
            data = np.broadcast_to(ent_vals, (reps, len(ent_vals)))
            df = pd.DataFrame(data, columns=list(evf.entities.keys()))
            index.append(df)
        self.dense_index = pd.concat(index, axis=0).reset_index(drop=True)

    def _densify_columns(self, cols):
        ''' Convert the named SparseBIDSColumns to DenseBIDSColumns.
        Args:
            cols (list): Column names to convert from sparse to dense.
        '''
        for c in cols:
            if isinstance(c, SparseBIDSColumn):
                self.collection[c] = self.collection[c].to_dense(self)

    def apply(self, func, cols, *args, **kwargs):
        ''' Applies an arbitrary callable or named function. Mostly useful for
        automating transformations via an external spec.
        Args:
            func (str, callable): ither a callable, or a string giving the
                name of an existing bound method to apply.
            args, kwargs: Optional positional and keyword arguments to pass
                on to the callable.
        '''
        if isinstance(func, string_types):
            func = globals()[func]
            func(self, cols, *args, **kwargs)

    def apply_from_json(self, spec):
        ''' Apply a series of transformations from a JSON spec.
        spec (str): Path to the JSON file containing transformations.
        '''
        if os.path.exists(spec):
            spec = json.load(open(spec, 'rU'))
        for t in spec['transformations']:
            name = t.pop('name')
            cols = t.pop('input', None)
            self.apply(name, cols, **t)
