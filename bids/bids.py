from __future__ import absolute_import, division, print_function

import os

from .due import due, Doi

__all__ = []

_SPECIAL_FILES = (
    'CHANGES',
    'ISSUES',
    'LICENSE',
    'README',
    'dataset_description.json',
)

_DATA_TYPES = (
    'func',
    'anat',
    'behav',
    'dwi',
    'fmap',
)
# Use duecredit (duecredit.org) to provide a citation to relevant work to
# be cited. This does nothing, unless the user has duecredit installed,
# And calls this with duecredit (as in `python -m duecredit script.py`):
due.cite(Doi("10.1038/sdata.2016.44"),
         description="Brain Imaging Data Structure",
         tags=["reference-implementation"],
         path='bids')


def _parse_filename(fname):
    components = fname.split('_')
    ftype = components[-1]
    components = dict([c.split('-') for c in components[:-1]])
    components['filetype'] = ftype
    return components


def _parse_dirname(dirs, components):
    for depth, dname in enumerate(dirs.split(os.path.sep)):
        if not dname:
            # catch top-level paths
            break
        if not depth and dname.startswith('sub-'):
            # subject-specific file
            if not dname.split('-')[1] == components['sub']:
                raise ValueError(
                    "inconsistent subject labels in '{0}'.format(fname)")
        if depth and dname.startswith('ses-'):
            # session-specific file
            if not dname.split('-')[1] == components['ses']:
                raise ValueError(
                    "inconsistent session labels in '{0}'.format(fname)")
        if depth:
            # must be a datatype (in the BIDS sense, e.g. 'func', 'anat', ...)
            components['datatype'] = dname
    return components


def parse_filename(fname):
    """Parse a BIDS filename and return a dict of its components

    Parameters
    ----------
    fname : str
      path to a file in a BIDS dataset, relative to the root of the dataset

    See also
    --------
    `assemble_filename()`

    Returns
    -------
    dict
      keys of the dictionary match the arguments of `assemble_filename()`
    """
    if not fname:
        return None

    dirs, fname = os.path.split(fname)

    components = None
    if dirs and not dirs.startswith('sub-'):
        # we are not dealing with anything but files in sub-* directories
        # here, capture path to be able to reconstruct original input
        components = dict(path=dirs, filetype=fname)
        return components
    if not dirs and fname in _SPECIAL_FILES:
        # top-level BIDS meta data
        components = dict(filetype=fname)
    else:
        components = _parse_filename(fname)

    # parse directory name and return
    return _parse_dirname(dirs, components)


def assemble_filename(filetype,
                      sub=None, ses=None, datatype=None, task=None, run=None,
                      recording=None):
    """Assemble a BIDS filename from its semantic components.

    See also
    --------

    This is the inverse of `assemble_filename()`.
    """
    pass
