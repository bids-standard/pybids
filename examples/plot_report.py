# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""

.. _reports1:

=========================================
 Report generation
=========================================

Generate a report for the provided test dataset.

"""
###############################################################################
# Start with the necessary imports
# --------------------------------
from textwrap import TextWrapper
from os.path import join
from bids.grabbids import BIDSLayout
from bids.reports import BIDSReport
from bids.tests import get_test_data_path

###############################################################################
# Define function for printing descriptions
# -----------------------------------------
def format_string(s, w=60):
    """
    Wraps long strings at some length while preserving newlines and tabs in
    original string.
    """
    lines = s.split('\n')
    new_s = ''
    for line in lines:
        if len(line) > w:
            wrapper = TextWrapper(width=w, break_long_words=False,
                                  replace_whitespace=False)
            line = '\n'.join(wrapper.wrap(line))
        new_s += line + '\n'
    return new_s

###############################################################################
# Generate report
# ----------------------------------
layout = BIDSLayout(join(get_test_data_path(), 'synthetic'))
report = BIDSReport(layout)
counter = report.generate(session='01')

###############################################################################
# Reformat report for notebook and print
# ---------------------------------------
description = counter.most_common()[0][0]
description = format_string(description, 80)
print(description)
