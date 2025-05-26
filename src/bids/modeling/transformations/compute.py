"""
Transformations that primarily involve numerical computation on variables.
"""
import math
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from bids.utils import listify
from .base import Transformation
from bids.modeling import hrf
from bids.variables import SparseRunVariable,  DenseRunVariable


def _fractional_gcd(vals, res=0.001):
    from functools import reduce
    from math import gcd
    return reduce(gcd, (int(np.round(val / res)) for val in vals)) * res


class Convolve(Transformation):
    """Convolve the input variable with an HRF.

    Parameters
    ----------
    var : Variable
        The variable to convolve.
    model : str
        The name of the HRF model to apply. Must be one of 'spm',
        'glover', or 'fir'.
    derivative : bool
        Whether or not to include the temporal derivative.
    dispersion : bool
        Whether or not to include the dispersion derivative.
    fir_delays : iterable
        A list or iterable of delays to use if model is
        'fir' (ignored otherwise). Spacing between delays must be fixed.

    Notes
    -----
    Uses the HRF convolution functions implemented in nilearn.
    """
    _groupable = False
    _input_type = 'variable'
    _return_type = 'variable'
    _sync_kwargs = False

    def _transform(self, var, model='spm', derivative=False, dispersion=False,
                   fir_delays=None):

        model = model.lower()

        df = var.to_df(entities=False)

        if isinstance(var, SparseRunVariable):
            sampling_rate = self.collection.sampling_rate
            dur = var.get_duration()
            resample_frames = np.linspace(
                0, dur, int(math.ceil(dur * sampling_rate)), endpoint=False)
            safety = 2  # Double frequency to resolve events
        else:
            resample_frames = df['onset'].values
            sampling_rate = var.sampling_rate
            safety = 1  # Maximum signal resolution is already 0.5 * SR

        vals = df[['onset', 'duration', 'amplitude']].values.T

        if model in ['spm', 'glover']:
            if derivative:
                model += ' + derivative'
            if dispersion:
                model += ' + dispersion'
        elif model != 'fir':
            raise ValueError("Model must be one of 'spm', 'glover', or 'fir'.")

        # Sampling at >100Hz will never be useful, but can be wildly expensive
        max_freq, min_interval = 100, 0.01
        # Sampling at <1Hz can degrade signals
        min_freq, max_interval = 1, 1

        # Given the sampling rate, determine an oversampling factor to ensure that
        # events can be modeled with reasonable precision
        unique_onsets = np.unique(df.onset)
        unique_durations = np.unique(df.duration)
        # Align existing data ticks with, event onsets and offsets, up to ms resolution
        # Note that GCD ignores zeros, so 0 onsets and impulse responses (0 durations) do
        # not harm this.
        required_resolution = _fractional_gcd(
            np.concatenate((unique_onsets, unique_durations)),
            res=min_interval)
        # Bound the effective sampling rate between min_freq and max_freq
        effective_sr = max(min_freq, min(safety / required_resolution, max_freq))
        convolved = hrf.compute_regressor(
            vals, model, resample_frames, fir_delays=fir_delays, min_onset=0,
            oversampling=int(np.ceil(effective_sr / sampling_rate))
            )

        results = []
        arr, names = convolved
        for conv, name in zip(np.split(arr, arr.shape[1], axis=1), names):
            new_name = '_'.join([var.name, name.split('_')[-1]]) if '_' in name else var.name
            results.append(
                DenseRunVariable(
                    name=new_name, values=conv, run_info=var.run_info,
                    source=var.source, sampling_rate=sampling_rate)
            )
        return results


class Demean(Transformation):

    def _transform(self, data):
        return data - data.mean()


class Orthogonalize(Transformation):

    _variables_used = ('variables', 'other')
    _densify = ('variables', 'other')
    _aligned_required = 'force_dense'
    _aligned_variables = ('other')
    _sync_kwargs = False

    def _transform(self, var, other):

        other = listify(other)

        # Set up X matrix and slice into it based on target variable indices
        X = np.array([self._variables[c].values.values.squeeze()
                      for c in other]).T
        X = X[var.index, :]
        assert len(X) == len(var)
        y = var.values
        _aX = np.c_[np.ones(len(y)), X]
        coefs, resids, rank, s = np.linalg.lstsq(_aX, y, rcond=None)
        result = pd.DataFrame(y - X.dot(coefs[1:]), index=var.index)
        return result


class Product(Transformation):

    _loopable = False
    _groupable = False
    _aligned_required = True
    _output_required = True

    def _transform(self, data):
        data = pd.concat(data, axis=1, sort=True)
        return data.product(axis=1)


class Scale(Transformation):
    """Scale a variable.

    Parameters
    ----------
    data : :obj:`pandas.Series` or :obj:`pandas.DataFrame`
        The variables to scale.
    demean : bool
        If True, demean each column.
    rescale : bool
        If True, divide variables by their standard deviation.
    replace_na : str
        Whether/when to replace missing values with 0. If
        None, no replacement is performed. If 'before', missing values are
        replaced with 0's before scaling. If 'after', missing values are
        replaced with 0 after scaling.

    Notes
    -----
    If a constant column is passed in, and replace_na is None or 'before', an
    exception will be raised.
    """

    def _transform(self, data, demean=True, rescale=True, replace_na=None):
        if data.nunique() == 1 and replace_na in {None, 'before'}:
            val = data.unique()[0]
            raise ValueError("Cannot scale a column with constant value ({})! "
                             "If you want a constant column of 0's returned, "
                             "set replace_na to 'after'.".format(val))
        if replace_na == 'before':
            data = data.fillna(0.)
        if demean:
            data -= data.mean()
        if rescale:
            data /= data.std()
        if replace_na == 'after':
            data = data.fillna(0.)
        return data


class Sum(Transformation):

    _loopable = False
    _groupable = False
    _aligned_required = True
    _output_required = True
    _sync_kwargs = False

    def _transform(self, data, weights=None):
        data = pd.concat(data, axis=1, sort=True)
        if weights is None:
            weights = np.ones(data.shape[1])
        else:
            weights = np.array(weights)
            if len(weights.ravel()) != data.shape[1]:
                raise ValueError("If weights are passed to sum(), the number "
                                 "of elements must equal number of variables"
                                 " being summed.")
        return (data * weights).sum(axis=1)


class Threshold(Transformation):
    """Threshold and/or binarize a variable.

    Parameters
    ----------
    data :obj:`pandas.Series` or :obj:`pandas.DataFrame`
        The pandas structure to threshold.
    threshold : float
        The value to binarize around (values above will
        be assigned 1, values below will be assigned 0).
    binarize : bool
        If True, binarizes all non-zero values (i.e., every
        non-zero value will be set to 1).
    above : bool
        Specifies which values to retain with respect to the
        cut-off. If True, all value above the threshold will be kept; if
        False, all values below the threshold will be kept. Defaults to
        True.
    signed : bool
        Specifies whether to treat the threshold as signed
        (default) or unsigned. For example, when passing above=True and
        threshold=3, if signed=True, all and only values above +3 would be
        retained. If signed=False, all absolute values > 3 would be retained
        (i.e.,values in  the range -3 < X < 3 would be set to 0).
    """

    _groupable = False

    def _transform(self, data, threshold=0., binarize=False, above=True,
                   signed=True):
        if not signed:
            threshold = np.abs(threshold)
            data = data.abs()
        keep = data >= threshold if above else data <= threshold
        data[~keep] = 0
        if binarize:
            data[keep] = 1
        return data


class And_(Transformation):
    """Logical AND on two or more variables.

    Parameters
    ----------
    dfs : list of :obj:`pandas.DataFrame`
        variables to enter into the conjunction.
    """

    _loopable = False
    _groupable = False
    _output_required = True
    _aligned_required = True
    _sync_kwargs = False

    def _transform(self, dfs):
        df = pd.concat(dfs, axis=1, sort=True)
        return df.all(axis=1).astype(int)


class Not(Transformation):
    """Logical negation of a variable.

    Parameters
    ----------
    var : :obj:`pandas.Series`
        Variable to negate. Must be convertible to bool.
    """

    _loopable = True
    _groupable = False
    sync_kwargs = False

    def _transform(self, var):
        return ~var.astype(bool)


class Or_(Transformation):
    """Logical OR (inclusive) on two or more variables.

    Parameters
    ----------
    dfs : list of :obj:`pandas.DataFrame`
        variables to enter into the disjunction.
    """

    _loopable = False
    _groupable = False
    _output_required = True
    _aligned_required = True
    sync_kwargs = False

    def _transform(self, dfs):
        df = pd.concat(dfs, axis=1, sort=True)
        return df.any(axis=1).astype(int)


class Lag(Transformation):
    """Lag variable

    Returns a variable that is lagged by a specified number of time points.
    Spline interpolation of the requested ``order`` is used for non-integer
    shifts.
    Points outside the input are filled according to the given ``mode``.
    Negative shifts move values toward the beginning of the sequence.
    If ``difference`` is ``True``, the backward difference is calculated for
    positive shifts, and the forward difference is calculated for negative
    shifts.

    Additional ``mode``s may be defined if there is need for them.
    The ``scipy.ndimage.shift`` method provides the reference implementation
    for all current modes. "Constant" is equivalent to the shift parameter
    "cval".


    Parameters
    ----------
    var : :obj:`numpy.ndarray`
        variable to lag
    shift : float, optional
        number of places to shift the values (default: 1)
    order : int, optional
        order of spline interpolation, from 0-5 (default: 3)
    mode : string
        the `mode` parameter determines how the input array is extended
        beyond its boundaries. Default is 'nearest'.
        The following values are accepted:

        "nearest" (a a a a | a b c d | d d d d)
            The input is extended by replicating the boundary values
        "reflect" (d c b a | a b c d | d c b a)
            The input is extended by reflecting the array over the edge
        "constant" (k k k k | a b c d | k k k k)
            The input is extended by filling all values beyond the edge
            with the same constant value, defined by the ``constant`` parameter
    constant : float, optional
        value to fill past edges of input if ``mode`` is ``"constant"``.
        (default: 0)
    difference : boolean, optional
        Calculate the backward (if shift is positive) or forward (if shift is
        negative) difference.
        For the forward difference dx of an array x, dx[i] = x[i+1] - x[i].
        For the backward difference dx of an array x, dx[i] = x[i] - x[i-1].
        (default: ``False``)
    """

    _input_type = 'numpy'
    _return_type = 'numpy'

    def _transform(self, var, shift=1, order=3, mode="nearest",
                   constant=0.0, difference=False):
        var = var.flatten()
        shifted = ndi.shift(var, shift=shift, order=order, mode=mode,
                            cval=constant)
        if not difference:
            return shifted
        elif shift >= 0:
            return var - shifted
        else:
            return shifted - var
