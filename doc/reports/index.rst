.. _reports:

=====================================================
``reports``: Data acquisition report generation
=====================================================

.. contents:: **Contents**
    :local:
    :depth: 1

|

.. currentmodule:: pybids.reports

.. _initializing_reports:

Initializing reports
===========================================

The :obj:`bids.reports.BIDSReport` class requires a :obj:`bids.BIDSLayout` object as an argument::

    >>> from os.path import join
    >>> from bids import BIDSLayout
    >>> from bids.reports import BIDSReport
    >>> from bids.tests import get_test_data_path
    >>> layout = BIDSLayout(join(get_test_data_path(), 'synthetic'))
    >>> report = BIDSReport(layout)

.. _generating_reports:

Generating reports
===========================================

Pybids reports are then generated with the ``generate`` method, which returns a
:obj:`collections.Counter` of reports::

    >>> counter = report.generate()
    >>> main_report = counter.most_common()[0][0]
    >>> print(main_report)
    r"""
    For session 01:
    	MR data were acquired using a UNKNOWN-Tesla MANUFACTURER MODEL MRI scanner.
    	Ten runs of N-Back UNKNOWN-echo fMRI data were collected (64 slices; repetition time, TR=2500ms;
    echo time, TE=UNKNOWNms; flip angle, FA=UNKNOWN<deg>; field of view, FOV=128x128mm;
    matrix size=64x64; voxel size=2x2x2mm). Each run was 2:40 minutes in length, during
    which 64 functional volumes were acquired.
    	Five runs of Rest UNKNOWN-echo fMRI data were collected (64 slices; repetition time, TR=2500ms;
    echo time, TE=UNKNOWNms; flip angle, FA=UNKNOWN<deg>; field of view, FOV=128x128mm;
    matrix size=64x64; voxel size=2x2x2mm). Each run was 2:40 minutes in length, during
    which 64 functional volumes were acquired.

    For session 02:
    	MR data were acquired using a UNKNOWN-Tesla MANUFACTURER MODEL MRI scanner.
    	Ten runs of N-Back UNKNOWN-echo fMRI data were collected (64 slices; repetition time, TR=2500ms;
    echo time, TE=UNKNOWNms; flip angle, FA=UNKNOWN<deg>; field of view, FOV=128x128mm;
    matrix size=64x64; voxel size=2x2x2mm). Each run was 2:40 minutes in length, during
    which 64 functional volumes were acquired.
    	Five runs of Rest UNKNOWN-echo fMRI data were collected (64 slices; repetition time, TR=2500ms;
    echo time, TE=UNKNOWNms; flip angle, FA=UNKNOWN<deg>; field of view, FOV=128x128mm;
    matrix size=64x64; voxel size=2x2x2mm). Each run was 2:40 minutes in length, during
    which 64 functional volumes were acquired.

    Dicoms were converted to NIfTI-1 format. This section was (in part) generated
    automatically using pybids (0.5)."""

.. _generating_subreports:

Generating reports on subsets of the data
-------------------------------------------

The ``generate`` method allows for keyword restrictions, just like
:obj:`bids.BIDSLayout`'s ``get`` method. For example, to
generate a report only for ``nback`` task data in session ``01``::

    >>> counter = report.generate(session='01', task='nback')
    >>> main_report = counter.most_common()[0][0]
    >>> print(main_report)
    r"""
    For session 01:
      MR data were acquired using a UNKNOWN-Tesla MANUFACTURER MODEL MRI scanner.
      Ten runs of N-Back fMRI data were collected (64 slices; repetition time,
    TR=2500ms; echo time, TE=UNKNOWNms; flip angle, FA=UNKNOWN<deg>; field of
    view, FOV=128x128mm; matrix size=64x64; voxel size=2x2x2mm). Each run was
    2:40 minutes in length, during which 64 functional volumes were acquired.

    Dicoms were converted to NIfTI-1 format. This section was (in part)
    generated automatically using pybids (0.5)."""


.. note::

    For a more detailed set of examples, please refer to the  Tutorial: :doc:`/examples/reports_tutorial`.