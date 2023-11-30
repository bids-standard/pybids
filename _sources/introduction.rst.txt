.. _introduction:

=====================================================
Introduction: What is pybids?
=====================================================

.. contents:: **Contents**
    :local:
    :depth: 1

What is BIDS and why should you use it?
========================================

.. topic:: **Why use BIDS?**

  The Brain Imaging Data Structure provides a standard for organizing, naming, and
  describing neuroimaging datasets. By organizing your data in this manner, you can
  make sure of a number of tools that are designed to work with BIDS datasets out
  of the box. Moreover, many open datasets are provided in BIDS format, which
  means that being familiar with this structure will make it easier for you to
  analyze public data as well as your own.

  For more information on the Brain Imaging Data Structure (BIDS), visit https://bids.neuroimaging.io.

What is pybids and why should you use it?
==========================================

.. topic:: **Why use pybids?**

  Pybids makes it easier for neuroimagers who utilize the BIDS standard to query,
  summarize, and manipulate their data. A number of Python packages for analyzing
  neuroimaging data, including `Nipype`_ and `nilearn`_, are optimized to
  work with BIDS datasets.

Installing pybids
=================
Pybids is easy to install.
To install the most recent release, use pip::

>>> pip install pybids

If you want the "bleeding-edge" version of pybids, you can install directly from
the GitHub repository::

>>> pip install git+https://github.com/bids-standard/pybids.git

Finding help
==============

:Mailing lists and forums:

    * Don't hesitate to ask questions about pybids on `NeuroStars
      <https://neurostars.org/tags/pybids>`_.

    * You can find help with neuroimaging in Python (file I/O,
      neuroimaging-specific questions) via the nipy user group:
      https://groups.google.com/forum/?fromgroups#!forum/nipy-user

    * If you notice a bug in the pybids code, please `open an issue`_ in the
      pybids repository.

.. _nilearn: https://nilearn.github.io
.. _Nipype: https://nipype.readthedocs.io
.. _open an issue: https://github.com/bids-standard/pybids/issues
