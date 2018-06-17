""" Tests of BIDS-specific functionality. Generic tests of core grabbit
functionality should go in the grabbit package. """

import pytest
from bids.grabbids import BIDSLayout
from os.path import join, abspath, basename
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
    data_dir = join(get_test_data_path(), 'ds005')
    return BIDSLayout([(data_dir, ['bids', 'derivatives'])], root=data_dir)


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


def test_get_metadata5(testlayout1):
    target = 'sub-01/ses-1/func/sub-01_ses-1_task-rest_acq-fullbrain_run-1_bold.nii.gz'
    result = testlayout1.get_metadata(join(testlayout1.root, target),
                                      include_entities=True)
    assert result['EchoTime'] == 0.020
    assert result['subject'] == '01'
    assert result['acquisition'] == 'fullbrain'


def test_get_events(testlayout3):
    target = ('sub-01/func/sub-01_task-'
              'mixedgamblestask_run-01_bold.nii.gz')
    result = testlayout3.get_events(join(testlayout3.root, target))
    assert len(result) == 2
    expected1 = abspath(join(
        testlayout3.root, target.replace('_bold.nii.gz', '_events.tsv')))
    assert expected1 in result

    expected2 = abspath(join(
        testlayout3.root, 'derivatives/events/', basename(expected1)))

    merged = testlayout3.get_events(join(testlayout3.root, target),
                                    return_type='df')

    assert 'response' in merged
    assert 'trial_type' in merged
    assert merged[merged.onset == 1].RT.values[0] == 1.0
    assert merged[merged.onset == 18].RT.values[0] == 100.0
    assert merged[merged.onset == 102].RT.values[0] > 1

    assert expected2 in result


def test_get_events2(testlayout2):
    target = 'sub-03/anat/sub-03_T1w.nii.gz'
    result = testlayout2.get_events(join(testlayout2.root, target))
    assert result is None


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


def test_layout_with_derivs(testlayout3):
    assert isinstance(testlayout3.files, dict)
    assert set(testlayout3.domains.keys()) == {'bids', 'derivatives'}
    assert testlayout3.domains['bids'].files
    assert testlayout3.domains['derivatives'].files
    assert 'derivatives.roi' in testlayout3.entities
    assert 'bids.roi' not in testlayout3.entities
    assert 'bids.subject' in testlayout3.entities


def test_layout_with_custom_domain_options():
    pass
