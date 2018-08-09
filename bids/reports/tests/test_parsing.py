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
    config_file = abspath(join(get_test_data_path(), '../../reports/config/converters.json'))
    with open(config_file, 'r') as fobj:
        config = json.load(fobj)
    return config

@pytest.fixture
def testmeta():
    metadata = {'RepetitionTime': 2.}
    return metadata

def test_parsing_anat():
    """
    parsing.anat_info returns a str description of each structural scan
    """
    type_ = 'T1w'
    metadata = testmeta()
    img = nib.load(join(get_test_data_path(), 'images/3d.nii.gz'))
    config = testconfig()
    desc = parsing.anat_info(type_, metadata, img, config)
    assert isinstance(desc, str)

def test_parsing_dwi():
    """
    parsing.dwi_info returns a str description of each diffusion scan
    """
    bval_file = join(get_test_data_path(), 'images/4d.bval')
    metadata = testmeta()
    img = nib.load(join(get_test_data_path(), 'images/4d.nii.gz'))
    config = testconfig()
    desc = parsing.dwi_info(bval_file, metadata, img, config)
    assert isinstance(desc, str)

def test_parsing_fmap():
    """
    parsing.fmap_info returns a str decsription of each field map
    """
    metadata = testmeta()
    metadata['PhaseEncodingDirection'] = 'j-'
    img = nib.load(join(get_test_data_path(), 'images/3d.nii.gz'))
    config = testconfig()
    layout = testlayout()
    desc = parsing.fmap_info(metadata, img, config, layout)
    assert isinstance(desc, str)

def test_parsing_func():
    """
    parsing.func_info returns a str description of a set of functional scans grouped
    by task
    """
    metadata = testmeta()
    img = nib.load(join(get_test_data_path(), 'images/4d.nii.gz'))
    config = testconfig()
    desc = parsing.func_info('nback', 3, metadata, img, config)
    assert isinstance(desc, str)

def test_parsing_genacq():
    """
    parsing.general_acquisition_info returns a str description of the scanner from
    minimal metadata
    """
    metadata = testmeta()
    desc = parsing.general_acquisition_info(metadata)
    assert isinstance(desc, str)

def test_parsing_final():
    """
    parsing.final_paragraph returns a str description of the dicom-to-nifti
    conversion process from minimal metadata
    """
    metadata = testmeta()
    desc = parsing.final_paragraph(metadata)
    assert isinstance(desc, str)

def test_parsing_parse():
    """
    parsing.parse_niftis should return a list of strings, with each string
    containing the description for a single nifti file (except functional data,
    which is combined within task, across runs)
    """
    layout = testlayout()
    subj = '01'
    niftis = layout.get(subject=subj, extensions='nii.gz')
    config = testconfig()
    desc = parsing.parse_niftis(layout, niftis, subj, config)
    assert isinstance(desc, list)
    assert isinstance(desc[0], str)
