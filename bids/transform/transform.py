import numpy as np
import pandas as pd
from .utils import listify
from .base import DenseBIDSColumn, SparseBIDSColumn
import warnings
import os
import json
import math
from six import string_types


def loopable(func):
    ''' A decorator that implicitly loops over the named columns in the first
    argument ('cols') to the decorated function. When calling the decorated
    function, the actual BIDSColumn is passed as the first argument (rather
    than the column name).

    Additionally, the 'output' argument in kwargs will
    be intercepted and used to determine whether to overwrite the existing
    column.

    Example:

    ```
    @loopable
    def widgetize(self, col, arg1, arg2):
        ...

    trans.widgetize(['col_a', 'col_b'], 4, 'apple', output=['a_000', 'b_000'])
    ```

    In the above, the wrapped widgetize() method will be called twice, with the
    first argument being the column in the current EventCollection that has the
    name 'col_a' and 'col_b', respectively. The resulting columns will then be
    assigned to the new keys 'a000' and 'b000', respectively (the original
    columns will be left untouched). If output is not explicitly set by the
    user, the result of the wrapped function will always overwrite the existing
    column.

    '''

    def wrapper(self, cols, *args, **kwargs):

        output = listify(kwargs.pop('output', None))
        cols = listify(cols)

        for i, c in enumerate(cols):

            c = self.collection.columns[c]

            result = func(self, c, *args, **kwargs)

            if output is None:
                c.values = result
            else:
                if len(cols) == len(output):
                    _output = output[i]
                elif len(output) == 1:
                    _output = str(output) + '_' + c
                self.collection[_output] = c.clone(data=result)

    return wrapper


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

    def _check_column_alignment(self, cols, force=True):
        ''' Checks whether the specified columns have aligned indexes. This
        implies either that all columns are dense, or that all columns are
        sparse and have exactly the same onsets and durations. If columns are
        not aligned and force = True, all columns will be forced to dense
        format in order to ensure alignment.
        '''

        # If any column is dense, all columns must be dense
        dense = [isinstance(c, DenseBIDSColumn) for c in cols]
        if any(dense):
            if not all(dense):
                sparse = [c for i, c in enumerate(cols) if not dense[i]]
                msg = ("Found a mix of dense and sparse columns when"
                       "attempting to orthogonalize variables.")
                if force:
                    msg += (" Sparse columns  %s will be converted to dense "
                            "form." % sparse)
                    for s in sparse:
                        self.collection[s] = self.collection[s].to_dense(self)
            warnings.warn(msg)

        # If all are sparse, durations and onsets must match perfectly for all
        else:
            def get_col_data(name):
                col = self.collection[name]
                return np.array([col.durations, col.onsets]).T
            # Compare 1st col with each of the others
            fc = get_col_data(cols[0])
            compare_cols = lambda a, b: (len(a) == len(b)) and np.allclose(a, b)
            if not all([compare_cols(fc, get_col_data(c)) for c in cols[1:]]):
                msg = "Misaligned sparse columns found."
                if force:
                    msg += (" Forcing all sparse columns to dense in order to "
                            "ensure proper alignment.")
                    for c in cols:
                        self.collection[c] = self.collection[c].to_dense(self)
                warnings.warn(msg)

    def apply(self, func, *args, **kwargs):
        ''' Applies an arbitrary callable or named function. Mostly useful for
        automating transformations via an external spec.
        Args:
            func (str, callable): Either a callable, or a string giving the
                name of an existing bound method to apply.
            args, kwargs: Optional positional and keyword arguments to pass
                on to the callable.
        '''
        if isinstance(func, string_types):
            func = getattr(self, func)
        func(*args, **kwargs)

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

    def rename(self, cols, output):
        ''' Rename one or more columns.

        Args:
            cols (str, list): Names of existing columns to rename.
            output (str, list): New names of columns.

        Details: If `cols` and `output` have the same number of elements, then
            named columns will be mapped from old to new names in a 1-to-1
            fashion. If there is only a single value in `output`, and more than
            1 value in `cols`, old column names will be prepended with the
            value in `output`.

        '''
        cols = listify(cols)
        output = listify(output)
        for _old, _new in dict(zip(cols, output)).items():
            self.collection.columns[_new] = self.collection.columns[_old]
            self.collection.columns.pop(_old)
            self.collection.columns[_new].name = _new

    @loopable
    def scale(self, col, demean=True, rescale=True):
        ''' Scale a column by subtracting its mean and/or dividing by its
        standard deviation. By default, this is equivalent to standardization.

        Args:
            col (BIDSColumn): The column to scale.
            demean (bool): Whether or not to subtract the column mean.
            rescale (bool): Whether or not to divide by the standard deviation.
        '''

        data = col.values.copy()
        if demean:
            data -= data.mean(0)
        if rescale:
            data /= data.std(0)

        return data

    @loopable
    def _binarize(self, col, threshold=0.0):
        ''' Binarize a column around a specified threshold.
        Args:
            col (BIDSColumn): The column to scale.
            threshold (float): The value to binarize around (values above will
                be assigned 1, values below will be assigned 0).
        '''
        data = col.values.copy()
        above = data > threshold
        data[above] = 1
        data[~above] = 0
        return data

    @loopable
    def orthogonalize(self, col, other, force_dense=False):
        ''' Orthgonalize a column with respect to one or more other columns.
        Args:
            col (BIDSColumn): The column to orthogonalize.
            other (list, str): The names of variables to orthogonalize the
                target column with respect to. Note that these must be strings,
                and not BIDSColumn instances.
            force_dense (bool): if True, all columns will be forced to dense
                format before orthogonalization, otherwise orthogonalization
                will be applied to the sparse representations (assuming that
                all columns are aligned properly).
        '''
        other = listify(other)

        if col.name in other:
            raise ValueError("Column %s cannot appear in both the set of "
                             "columns to orthogonalize and the set of columns "
                             "to orthogonalize with respect to!" % col.name)

        all_cols = [col.name] + other

        if force_dense:
            self._densify_columns(all_cols)

        self._check_column_alignment(all_cols, force=True)

        y = self.collection[col.name].values.values
        X = np.c_[[self.collection[c].values.values for c in other]]
        X = X.squeeze(axis=0)
        _aX = np.c_[np.ones(len(y)), X]
        coefs, resids, rank, s = np.linalg.lstsq(_aX, y)
        return y - X.dot(coefs[1:])
