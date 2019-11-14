"""
tests for bids.reports.parsing
"""
import json
from os.path import abspath, join
import pytest

import nibabel as nib

from bids.reports import parsing
from bids.layout import BIDSLayout
from bids.tests import get_test_data_path


@pytest.fixture
def testlayout():
    data_dir = join(get_test_data_path(), 'synthetic')
    return BIDSLayout(data_dir)


@pytest.fixture
def testconfig():
    config_file = abspath(join(get_test_data_path(),
                               '../../reports/config/converters.json'))
    with open(config_file, 'r') as fobj:
        config = json.load(fobj)
    return config


@pytest.fixture
def testmeta():
    metadata = {'RepetitionTime': 2.}
    return metadata


def test_parsing_anat(testmeta, testconfig):
    """
    parsing.anat_info returns a str description of each structural scan
    """
    type_ = 'T1w'
    img = nib.load(join(get_test_data_path(), 'images/3d.nii.gz'))
    desc = parsing.anat_info(type_, testmeta, img, testconfig)
    assert isinstance(desc, str)


def test_parsing_dwi(testmeta, testconfig):
    """
    parsing.dwi_info returns a str description of each diffusion scan
    """
    bval_file = join(get_test_data_path(), 'images/4d.bval')
    img = nib.load(join(get_test_data_path(), 'images/4d.nii.gz'))
    desc = parsing.dwi_info(bval_file, testmeta, img, testconfig)
    assert isinstance(desc, str)


def test_parsing_fmap(testlayout, testmeta, testconfig):
    """
    parsing.fmap_info returns a str decsription of each field map
    """
    testmeta['PhaseEncodingDirection'] = 'j-'
    img = nib.load(join(get_test_data_path(), 'images/3d.nii.gz'))
    desc = parsing.fmap_info(testmeta, img, testconfig, testlayout)
    assert isinstance(desc, str)


def test_parsing_func(testmeta, testconfig):
    """
    parsing.func_info returns a str description of a set of functional scans
    grouped by task
    """
    img = nib.load(join(get_test_data_path(), 'images/4d.nii.gz'))
    desc = parsing.func_info('nback', 3, testmeta, img, testconfig)
    assert isinstance(desc, str)


def test_parsing_genacq(testmeta):
    """
    parsing.general_acquisition_info returns a str description of the scanner
    from minimal metadata
    """
    desc = parsing.general_acquisition_info(testmeta)
    assert isinstance(desc, str)


def test_parsing_final(testmeta):
    """
    parsing.final_paragraph returns a str description of the dicom-to-nifti
    conversion process from minimal metadata
    """
    desc = parsing.final_paragraph(testmeta)
    assert isinstance(desc, str)


def test_parsing_parse(testlayout, testconfig):
    """
    parsing.parse_niftis should return a list of strings, with each string
    containing the description for a single nifti file (except functional data,
    which is combined within task, across runs)
    """
    subj = '01'
    niftis = testlayout.get(subject=subj, extension=[".nii", ".nii.gz"])
    desc = parsing.parse_niftis(testlayout, niftis, subj, testconfig)
    assert isinstance(desc, list)
    assert isinstance(desc[0], str)
