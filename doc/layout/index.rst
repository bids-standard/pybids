.. _layout:

=====================================================
``layout``: Querying BIDS datasets
=====================================================

.. contents:: **Contents**
    :local:
    :depth: 1

|

.. currentmodule:: pybids.layout

.. _loading_datasets:

Loading BIDS datasets
===========================================

The :obj:`bids.layout.BIDSLayout` class requires the path to a valid BIDS dataset::

    >>> from os.path import join
    >>> from bids import BIDSLayout
    >>> from bids.tests import get_test_data_path
    >>> layout = BIDSLayout(join(get_test_data_path(), 'synthetic'))

The ``BIDSLayout`` instance is a lightweight container for all of the files in the
BIDS project directory. It automatically detects any BIDS entities found in the
file paths, and allows us to perform simple but relatively powerful queries over
the file tree. By default, defined BIDS entities include things like "subject",
"session", "run", and "type".

.. hint::

    To exclude folders from indexing (e.g. only index a subset of subjects to save time),
    you can pass a list of folder names, or a regex pattern to the ``ignore`` argument. 
    For example, to ignore all subjects except "25":

        >>> BIDSLayout(bids_dir, ignore=[re.compile(r"(sub-(?!25)\d*/)")])

.. _querying_datasets:

Querying datasets
===========================================

Pybids layouts can be queried according to a number of parameters, using
grabbit's ``get`` functionality.

For example, if we want to get a list of subjects in the dataset::

    >>> layout.get_subjects()
    ['01', '02', '03', '04', '05']

We can also get a list of all available sessions::

    >>> layout.get_sessions()
    ['01', '02']

Or a list of tasks::

    >>> layout.get_tasks()
    ['nback', 'rest']

.. extracting_metadata:

Extracting metadata
====================

A number of ``BIDSLayout`` methods extract metadata associated with files.

For example, if we want event (task timing) information for a given fMRI scan, we can use ``get_events``::

    >>> f = layout.get(task='nback', run=1, extension='nii.gz')[0].filename
    >>> layout.get_events(f)

We can also extract metadata from the json files associated with a scan file::

    >>> f = layout.get(task='nback', run=1, extension='nii.gz')[0].filename
    >>> layout.get_metadata(f)


.. note::

    For a more detailed set of examples, please refer to the  Tutorial: :doc:`/examples/pybids_tutorial`
