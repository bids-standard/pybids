""" Tests of BIDS-specific functionality. Generic tests of core grabbit
functionality should go in the grabbit package. """

import pytest
from bids.grabbids import BIDSValidator
from os.path import join, dirname, abspath

# Fixture uses in the rest of the tests
@pytest.fixture
def testvalidator():
    return BIDSValidator()


def test_is_anat_true(testvalidator):
    target_list = ["/sub-01/anat/sub-01_T1w.json",
    "/sub-01/anat/sub-01_T1w.nii.gz",
    "/sub-01/anat/sub-01_rec-CSD_T1w.json",
    "/sub-01/anat/sub-01_rec-CSD_T1w.nii.gz",
    "/sub-01/anat/sub-01_acq-23_T1w.json",
    "/sub-01/anat/sub-01_acq-23_T1w.nii.gz",
    "/sub-01/anat/sub-01_acq-23_rec-CSD_T1w.json",
    "/sub-01/anat/sub-01_acq-23_rec-CSD_T1w.nii.gz",
    "/sub-01/anat/sub-01_run-23_T1w.json",
    "/sub-01/anat/sub-01_run-23_T1w.nii.gz",
    "/sub-01/anat/sub-01_rec-CSD_run-23_T1w.json",
    "/sub-01/anat/sub-01_rec-CSD_run-23_T1w.nii.gz",
    "/sub-01/anat/sub-01_acq-23_run-23_T1w.json",
    "/sub-01/anat/sub-01_acq-23_run-23_T1w.nii.gz",
    "/sub-01/anat/sub-01_acq-23_rec-CSD_run-23_T1w.json",
    "/sub-01/anat/sub-01_acq-23_rec-CSD_run-23_T1w.nii.gz",
    "/sub-01/ses-test/anat/sub-01_ses-test_T1w.json",
    "/sub-01/ses-test/anat/sub-01_ses-test_T1w.nii.gz",
    "/sub-01/ses-test/anat/sub-01_ses-test_rec-CSD_T1w.json",
    "/sub-01/ses-test/anat/sub-01_ses-test_rec-CSD_T1w.nii.gz",
    "/sub-01/ses-test/anat/sub-01_ses-test_acq-23_T1w.json",
    "/sub-01/ses-test/anat/sub-01_ses-test_acq-23_T1w.nii.gz",
    "/sub-01/ses-test/anat/sub-01_ses-test_acq-23_rec-CSD_T1w.json",
    "/sub-01/ses-test/anat/sub-01_ses-test_acq-23_rec-CSD_T1w.nii.gz",
    "/sub-01/ses-test/anat/sub-01_ses-test_run-23_T1w.json",
    "/sub-01/ses-test/anat/sub-01_ses-test_run-23_T1w.nii.gz",
    "/sub-01/ses-test/anat/sub-01_ses-test_rec-CSD_run-23_T1w.json",
    "/sub-01/ses-test/anat/sub-01_ses-test_rec-CSD_run-23_T1w.nii.gz",
    "/sub-01/ses-test/anat/sub-01_ses-test_acq-23_run-23_T1w.json",
    "/sub-01/ses-test/anat/sub-01_ses-test_acq-23_run-23_T1w.nii.gz",
    "/sub-01/ses-test/anat/sub-01_ses-test_acq-23_rec-CSD_run-23_T1w.json",
    "/sub-01/ses-test/anat/sub-01_ses-test_acq-23_rec-CSD_run-23_T1w.nii.gz"]

    for item in target_list:
        result = testvalidator.is_anat(item)
        if result != True:
            print(item)
        assert result == True

def test_is_anat_false(testvalidator):
    target_list = ["/sub-01/anat/sub-1_T1w.json", #subject incosistency
    "/sub-01/anat/sub-01_dwi.nii.gz", #wrong modality suffix
    "/sub-01/anat/sub-02_rec-CSD_T1w.json", #subject incosistency
    "/sub-01/anat/sub-01_rec-CS-D_T1w.nii.gz", #rec label wrong
    "/sub-01/anat/sub-01_acq-23_T1W.json", #modality suffix wrong
    "/sub-01/dwi/sub-01_acq-23_dwi.nii.gz", #wrong data type
    "/sub-01/anat/sub-01_acq-23_rec-CSD_T1w.exe", #wrong extension
    "/sub-01/anat/sub-01_acq-23_rec-CSD_T1w.niigz", #extension typo
    "/sub-01/anat/sub-01_run-2-3_T1w.json", #run label typo
    "/sub-01/anat/sub-01_rn-23_T1w.nii.gz", #run typo
    "/sub-01/ant/sub-01_rec-CS-D_run-23_T1w.json", #reconstruction label typo
    "/sub-1/anat/sub-01_rec-CSD_run-23_t1w.nii.gz", #T1w suffix typo
    "/sub-01/anat/sub-01_aq-23_run-23_T1w.json", # acq typo
    "/sub-01/anat/sub-01_acq-23_run-23_dwi.nii.gz", # wrong data type
    "/sub-01/anat/sub-01_acq-23_rc-CSD_run-23_T1w.json", #rec typo
    "/sub-01/anat/sub-O1_acq-23_rec-CSD_run-23_T1w.nii.gz", #2nd subject id typo
    "/sub-01/ses-test/anat/sub-01_ses-retest_T1w.json", #ses inconsistency
    "/sub-01/ses-test/anat/sub-01_sestest_T1w.nii.gz", #2nd session typo
    "/sub-01/ses-test/anat/sub-01_ses_test_rec-CSD_dwi.jsn", #extension typo
    "/sub-01/ses_test/anat/sub-01_ses_test_rec-CSD_T1w.bval", #wrong extension
    "/sub-01/ses-test/anat/sub-01_ses-test_acq-23_T1w.exe",  #wrong extension
    ]
    for item in target_list:
        result = testvalidator.is_anat(item)
        if result != False:
            print(item)
        assert result == False
