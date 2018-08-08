""" Tests of BIDS-specific functionality. Generic tests of core grabbit
functionality should go in the grabbit package. """

import pytest
from bids.layout import BIDSLayout
from os.path import join, abspath
from bids.tests import get_test_data_path


# Fixture uses in the rest of the tests
@pytest.fixture(scope='module')
def testlayout1():
    data_dir = join(get_test_data_path(), '7t_trt')
    return BIDSLayout(data_dir)


@pytest.fixture(scope='module')
def testlayout2():
    data_dir = join(get_test_data_path(), 'ds005')
    return BIDSLayout(data_dir, exclude=['models/', 'derivatives/'])


@pytest.fixture(scope='module')
def testlayout3():
    data_dir = join(get_test_data_path(), 'ds000117')
    return BIDSLayout(data_dir)


@pytest.fixture(scope='module')
def deriv_layout():
    data_dir = join(get_test_data_path(), 'ds005')
    deriv_dir = join(data_dir, 'derivatives')
    return BIDSLayout([(data_dir, 'bids'),
                       (deriv_dir, ['bids', 'derivatives'])])


def test_layout_init(testlayout1):
    assert isinstance(testlayout1.files, dict)


def test_load_description(testlayout1):
    # Should not raise an error
    assert hasattr(testlayout1, 'description')
    assert testlayout1.description['Name'] == '7t_trt'
    assert testlayout1.description['BIDSVersion'] == "1.0.0rc3"


def test_get_metadata(testlayout1):
    target = 'sub-03/ses-2/func/sub-03_ses-2_task-' \
             'rest_acq-fullbrain_run-2_bold.nii.gz'
    result = testlayout1.get_metadata(join(testlayout1.root, target))
    assert result['RepetitionTime'] == 3.0


def test_get_metadata2(testlayout1):
    target = 'sub-03/ses-1/fmap/sub-03_ses-1_run-1_phasediff.nii.gz'
    result = testlayout1.get_metadata(join(testlayout1.root, target))
    assert result['EchoTime1'] == 0.006


def test_get_metadata3(testlayout1):
    target = 'sub-01/ses-1/func/sub-01_ses-1_task-rest_acq-fullbrain_run-1_bold.nii.gz'
    result = testlayout1.get_metadata(join(testlayout1.root, target))
    assert result['EchoTime'] == 0.020

    target = 'sub-01/ses-1/func/sub-01_ses-1_task-rest_acq-fullbrain_run-2_bold.nii.gz'
    result = testlayout1.get_metadata(join(testlayout1.root, target))
    assert result['EchoTime'] == 0.017


def test_get_metadata4(testlayout2):
    target = 'sub-03/anat/sub-03_T1w.nii.gz'
    result = testlayout2.get_metadata(join(testlayout2.root, target))
    assert result == {}


def test_get_metadata_meg(testlayout3):
    funcs = ['get_subjects', 'get_sessions', 'get_tasks', 'get_runs', 'get_acqs', 'get_procs']
    assert all([hasattr(testlayout3, f) for f in funcs])
    procs = testlayout3.get_procs()
    assert procs == ['sss']
    target = 'sub-02/ses-meg/meg/sub-02_ses-meg_task-facerecognition_run-01_meg.fif.gz'
    result = testlayout3.get_metadata(join(testlayout3.root, target))
    metadata_keys = ['MEGChannelCount', 'SoftwareFilters', 'SubjectArtefactDescription']
    assert all([k in result for k in metadata_keys])

def test_get_metadata5(testlayout1):
    target = 'sub-01/ses-1/func/sub-01_ses-1_task-rest_acq-fullbrain_run-1_bold.nii.gz'
    result = testlayout1.get_metadata(join(testlayout1.root, target),
                                      include_entities=True)
    assert result['EchoTime'] == 0.020
    assert result['subject'] == '01'
    assert result['acquisition'] == 'fullbrain'


def test_get_bvals_bvecs(testlayout2):
    dwifile = testlayout2.get(subject="01", modality="dwi")[0]
    result = testlayout2.get_bval(dwifile.filename)
    assert result == abspath(join(testlayout2.root, 'dwi.bval'))

    result = testlayout2.get_bvec(dwifile.filename)
    assert result == abspath(join(testlayout2.root, 'dwi.bvec'))


def test_get_subjects(testlayout1):
    result = testlayout1.get_subjects()
    predicted = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
    assert predicted == result


def test_get_fieldmap(testlayout1):
    target = 'sub-03/ses-1/func/sub-03_ses-1_task-' \
             'rest_acq-fullbrain_run-1_bold.nii.gz'
    result = testlayout1.get_fieldmap(join(testlayout1.root, target))
    assert result["type"] == "phasediff"
    assert result["phasediff"].endswith('sub-03_ses-1_run-1_phasediff.nii.gz')


def test_get_fieldmap2(testlayout1):
    target = 'sub-03/ses-2/func/sub-03_ses-2_task-' \
             'rest_acq-fullbrain_run-2_bold.nii.gz'
    result = testlayout1.get_fieldmap(join(testlayout1.root, target))
    assert result["type"] == "phasediff"
    assert result["phasediff"].endswith('sub-03_ses-2_run-2_phasediff.nii.gz')


def test_bids_json(testlayout1):
    assert testlayout1.get(return_type='id', target='run') == ['1', '2']
    assert testlayout1.get(return_type='id', target='session') == ['1', '2']


def test_exclude(testlayout2):
    assert join(
        testlayout2.root, 'models/ds-005_type-russ_sub-all_model.json') \
        not in testlayout2.files
    assert 'all' not in testlayout2.get_subjects()
    for f in testlayout2.files.values():
        assert 'derivatives' not in f.path


def test_layout_with_derivs(deriv_layout):
    assert deriv_layout.root == join(get_test_data_path(), 'ds005')
    assert isinstance(deriv_layout.files, dict)
    assert set(deriv_layout.domains.keys()) == {'bids', 'derivatives'}
    assert deriv_layout.domains['bids'].files
    assert deriv_layout.domains['derivatives'].files
    assert 'derivatives.roi' in deriv_layout.entities
    assert 'bids.roi' not in deriv_layout.entities
    assert 'bids.subject' in deriv_layout.entities


def test_query_derivatives(deriv_layout):
    result = deriv_layout.get(type='events', return_type='object',
                              domains='derivatives')
    assert len(result) == 1
    assert result[0].filename == 'sub-01_task-mixedgamblestask_run-01_events.tsv'
