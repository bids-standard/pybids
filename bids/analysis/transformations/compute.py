'''
Transformations that primarily involve numerical computation on variables.
'''

import numpy as np
import pandas as pd
from bids.utils import listify
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
    _output_required = True

    def _transform(self, data, weights=None):
        if weights is None:
            weights = np.ones(data.shape[1])
        else:
            if len(weights.ravel()) != data.shape[1]:
                raise ValueError("If weights are passed to sum(), the number "
                                 "of elements must equal number of variables"
                                 "being summed.")
        data = pd.concat(data, axis=1)
        return data.dot(weights)


class product(Transformation):

    _loopable = False
    _groupable = False
    _align = True
    _output_required = True

    def _transform(self, data):
        data = pd.concat(data, axis=1)
        return data.product(1)


class orthogonalize(Transformation):

    _variables_used = ('variables', 'other')
    _densify = ('variables', 'other')
    _align = ('other')

    def _transform(self, var, other):

        other = listify(other)

        # Set up X matrix and slice into it based on target variable indices
        X = np.array([self._variables[c].values.values.squeeze()
                      for c in other]).T
        X = X[var.index, :]
        assert len(X) == len(var)
        y = var.values
        _aX = np.c_[np.ones(len(y)), X]
        coefs, resids, rank, s = np.linalg.lstsq(_aX, y)
        result = pd.DataFrame(y - X.dot(coefs[1:]), index=var.index)
        return result


class threshold(Transformation):
    ''' Threshold and/or binarize a variable.
    Args:
        data (Series/DF): The pandas structure to threshold.
        threshold (float): The value to binarize around (values above will
            be assigned 1, values below will be assigned 0).
        binarize (bool): If True, binarizes all non-zero values (i.e., every
            non-zero value will be set to 1).
        above (bool): Specifies which values to retain with respect to the
            cut-off. If True, all value above the threshold will be kept; if
            False, all values below the threshold will be kept. Defaults to
            True.
        signed (bool): Specifies whether to treat the threshold as signed
        (default) or unsigned. For example, when passing above=True and
        threshold=3, if signed=True, all and only values above +3 would be
        retained. If signed=False, all absolute values > 3 would be retained
        (i.e.,values in  the range -3 < X < 3 would be set to 0).

    '''

    _groupable = False

    def _transform(self, data, threshold=0., binarize=False, above=True,
                   signed=True):
        if not signed:
            threshold = np.abs(threshold)
            data = data.abs()
        keep = data >= threshold if above else data <= threshold
        # print("Keep:", keep)
        data[~keep] = 0
        if binarize:
            data[keep] = 1
        return data


class or_(Transformation):
    ''' Logical OR (inclusive) on two or more variables.
    Args:
        dfs (list of DFs): variables to enter into the disjunction.
    '''

    _loopable = False
    _groupable = False
    _output_required = True

    def _transform(self, dfs):
        df = pd.concat(dfs, axis=1)
        return df.any(axis=1).astype(int)


class and_(Transformation):
    ''' Logical AND on two or more variables.
    Args:
        dfs (list of DFs): variables to enter into the conjunction.
    '''

    _loopable = False
    _groupable = False
    _output_required = True

    def _transform(self, dfs):
        df = pd.concat(dfs, axis=1)
        return df.all(axis=1).astype(int)
