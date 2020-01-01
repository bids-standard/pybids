"""Generate publication-quality data acquisition methods section from BIDS dataset.

Utilities to generate the MRI data acquisition portion of a
methods section from a BIDS dataset.
"""
import logging
import os
from .. import __version__

logging.basicConfig()
LOGGER = logging.getLogger('pybids.reports.utils')


def reminder():
    """
    Remind users about things they need to do after generating the report.
    """
    return('Remember to double-check everything and to replace <deg> with '
           'a degree symbol.')


def remove_duplicates(seq):
    """
    Return unique elements from list while preserving order.
    From https://stackoverflow.com/a/480227/2589328
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def num_to_str(num):
    """
    Convert an int or float to a nice string.
    E.g.,
        21 -> '21'
        2.500 -> '2.5'
        3. -> '3'
    """
    return '{0:0.02f}'.format(num).rstrip('0').rstrip('.')


def list_to_str(lst):
    """
    Turn a list into a comma- and/or and-separated string.

    Parameters
    ----------
    lst : :obj:`list`
        A list of strings to join into a single string.

    Returns
    -------
    str_ : :obj:`str`
        A string with commas and/or ands separating the elements from ``lst``.

    """
    if len(lst) == 1:
        str_ = lst[0]
    elif len(lst) == 2:
        str_ = ' and '.join(lst)
    elif len(lst) > 2:
        str_ = ', '.join(lst[:-1])
        str_ += ', and {0}'.format(lst[-1])
    else:
        raise ValueError('List of length 0 provided.')
    return str_


def get_slice_info(slice_times):
    """
    Extract slice order from slice timing info.

    TODO: Be more specific with slice orders.
    Currently anything where there's some kind of skipping is interpreted as
    interleaved of some kind.

    Parameters
    ----------
    slice_times : array-like
        A list of slice times in seconds or milliseconds or whatever.

    Returns
    -------
    slice_order_name : :obj:`str`
        The name of the slice order sequence.
    """
    # Slice order
    slice_times = remove_duplicates(slice_times)
    slice_order = sorted(range(len(slice_times)), key=lambda k: slice_times[k])
    if slice_order == range(len(slice_order)):
        slice_order_name = 'sequential ascending'
    elif slice_order == reversed(range(len(slice_order))):
        slice_order_name = 'sequential descending'
    elif slice_order[0] < slice_order[1]:
        # We're allowing some wiggle room on interleaved.
        slice_order_name = 'interleaved ascending'
    elif slice_order[0] > slice_order[1]:
        slice_order_name = 'interleaved descending'
    else:
        slice_order = [str(s) for s in slice_order]
        raise Exception('Unknown slice order: [{0}]'.format(', '.join(slice_order)))

    return slice_order_name


def get_seqstr(config, metadata):
    """
    Extract and reformat imaging sequence(s) and variant(s) into pretty
    strings.

    Parameters
    ----------
    config : :obj:`dict`
        A dictionary with relevant information regarding sequences, sequence
        variants, phase encoding directions, and task names.
    metadata : :obj:`dict`
        The metadata for the scan.

    Returns
    -------
    seqs : :obj:`str`
        Sequence names.
    variants : :obj:`str`
        Sequence variant names.
    """
    seq_abbrs = metadata.get('ScanningSequence', '').split('_')
    seqs = [config['seq'].get(seq, seq) for seq in seq_abbrs]
    variants = [config['seqvar'].get(var, var) for var in \
                metadata.get('SequenceVariant', '').split('_')]
    seqs = list_to_str(seqs)
    if seq_abbrs[0]:
        seqs += ' ({0})'.format(os.path.sep.join(seq_abbrs))
    variants = list_to_str(variants)
    return seqs, variants


def get_sizestr(img):
    """
    Extract and reformat voxel size, matrix size, field of view, and number of
    slices into pretty strings.

    Parameters
    ----------
    img : :obj:`nibabel.Nifti1Image`
        Image from scan from which to derive parameters.

    Returns
    -------
    n_slices : :obj:`int`
        Number of slices.
    voxel_size : :obj:`str`
        Voxel size string (e.g., '2x2x2')
    matrix_size : :obj:`str`
        Matrix size string (e.g., '128x128')
    fov : :obj:`str`
        Field of view string (e.g., '256x256')
    """
    n_x, n_y, n_slices = img.shape[:3]
    import numpy as np
    voxel_dims = np.array(img.header.get_zooms()[:3])
    matrix_size = '{0}x{1}'.format(num_to_str(n_x), num_to_str(n_y))
    voxel_size = 'x'.join([num_to_str(s) for s in voxel_dims])
    fov = [n_x, n_y] * voxel_dims[:2]
    fov = 'x'.join([num_to_str(s) for s in fov])
    return n_slices, voxel_size, matrix_size, fov
