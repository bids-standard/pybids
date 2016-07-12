from __future__ import absolute_import, division, print_function

import pytest
import os.path as op
import bids
from bids.bids import parse_filename, assemble_filename

data_path = op.join(bids.__path__[0], 'data')


def test_parse_filename():
    # get nothing from nothing
    assert parse_filename(None) is None
    assert parse_filename('') is None

    # fail on inconsistency
    with pytest.raises(ValueError):
        parse_filename("sub-01/func/sub-02_task-some_bold.nii.gz")

    # special file handling (top-level meta data)
    for toplevelfile in ("dataset_description.json", 'ISSUES'):
        assert parse_filename(toplevelfile) == dict(filetype=toplevelfile)

    # top-level files that still need parsing
    assert parse_filename('task-objectcategories_bold.json') \
        == dict(task='objectcategories', filetype='bold.json')

    # things it doesn't know anything about
    assert parse_filename('tests/mybesttest.py') \
        == dict(path='tests', filetype='mybesttest.py')

    # full-scale test
    r = parse_filename("sub-03/ses-movie/func/sub-03_ses-movie_task-movie_run-4_recording-cardresp_physio.tsv.gz")
    assert r == dict(
        sub='03', ses='movie', datatype='func', task='movie', run='4',
        recording='cardresp', filetype='physio.tsv.gz')


def test_assemble_filename():
    # plain top-level files are simple
    assert assemble_filename('README') == 'README'
    assert assemble_filename('UNknown') == 'UNknown'
    # constructed top-level
    assert assemble_filename('bold.json', task='some') == 'task-some_bold.json'
    # can deal with unforeseen filenames/types and still respects the folder structure
    assert assemble_filename('stupid', sub='test', ses='best', datatype='extreme') \
        == 'sub-test/ses-best/extreme/sub-test_ses-best_stupid'
    # can deal with unforeseen datatypes
    assert assemble_filename('stupid', sub='test', ses='best') \
        == 'sub-test/ses-best/sub-test_ses-best_stupid'

    # round-trip
    fname = "sub-03/ses-movie/func/sub-03_ses-movie_task-movie_run-4_recording-cardresp_physio.tsv.gz"
    assert assemble_filename(**parse_filename(fname)) == fname
