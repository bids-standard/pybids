'''
Transformations that primarily involve numerical computation on columns.
'''

import numpy as np
import pandas as pd
from bids.events.utils import listify
from .base import Transformation


class scale(Transformation):

    def _transform(self, data, demean=True, rescale=True):
        if demean:
            data -= data.mean()
        if rescale:
            data /= data.std()
        return data


class sum(Transformation):

    _loopable = False
    _groupable = False
    _align = True

    def _transform(self, data, weights=None):
        if weights is None:
            weights = np.ones(data.shape[1])
        else:
            if len(weights.ravel()) != data.shape[1]:
                raise ValueError("If weights are passed to sum(), the number "
                                 "of elements must equal the number of columns"
                                 "being summed.")
        data = pd.concat(data, axis=1)
        return data.dot(weights)


class product(Transformation):

    _loopable = False
    _groupable = False
    _align = True

    def _transform(self, data):
        data = pd.concat(data, axis=1)
        return data.product(1)


class orthogonalize(Transformation):

    _columns_used = ('cols', 'other')
    _densify = ('cols', 'other')
    _align = ('other')

    def _transform(self, col, other):

        other = listify(other)

        # Set up X matrix and slice into it based on target column indices
        X = np.array([self._columns[c].values.values.squeeze()
                      for c in other]).T
        X = X[col.index, :]
        assert len(X) == len(col)
        y = col.values
        _aX = np.c_[np.ones(len(y)), X]
        coefs, resids, rank, s = np.linalg.lstsq(_aX, y)
        result = pd.DataFrame(y - X.dot(coefs[1:]), index=col.index)
        return result


class binarize(Transformation):
    ''' Binarize a column.
    Args:
        col (Series/DF): The pandas structure to binarize.
        threshold (float): The value to binarize around (values above will
            be assigned 1, values below will be assigned 0).
    '''

    _groupable = False
    _input_type = 'pandas'

    def _transform(self, data, threshold=0.):
        above = data > threshold
        data[above] = 1
        data[~above] = 0
        return data
