from __future__ import absolute_import, division, print_function

import os
from os.path import join as opj

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
                      recording=None, path=None):
    """Assemble a BIDS filename from its semantic components.

    Parameters
    ----------
    filetype : str
      Filename or file type labels, such as 'events.tsv', or 'bold.nii.gz'
    sub : str
      Subject label
    ses : str
      Session label
    datatype : str
      BIDS data type labels, such as 'func', 'anat', 'dwi', 'behav'.
      Unknown labels are supported.
    task : str
      Task label
    run : str
      Run label
    recording : str
      Recording label
    path : str
      If this is given, the function will return the `filetype` appended to
      this path (with a path separator). This enables support for non-BIDS
      dataset content.

    See also
    --------

    This is the inverse of `assemble_filename()`.
    """
    if path:
        # signal that we know very very little
        return opj(path, filetype)
    if sub is not None:
        # this isn't top-lebel stuff -> we need to build the directory
        dcomps = [('sub', sub)]
        if ses:
            dcomps.append(('ses', ses))
        if datatype:
            dcomps.append(datatype)
        # we can recycle 'path' now
        path = opj(*['-'.join(dc) if isinstance(dc, tuple) else dc for dc in dcomps])
    # assemble the filename
    vars = locals()
    fcomps = ['-'.join((c, vars[c]))
              for c in ('sub', 'ses', 'task', 'run', 'recording') if vars[c]]
    fcomps.append(filetype)
    fname = '_'.join(fcomps)
    if path:
        return opj(path, fname)
    else:
        return fname

    pass
