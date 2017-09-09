""" Tests of BIDS-specific functionality. Generic tests of core grabbit
functionality should go in the grabbit package. """

import pytest
from bids.grabbids import BIDSValidator
from os.path import join, dirname, abspath

# Fixture uses in the rest of the tests
@pytest.fixture
def testvalidator():
    return BIDSValidator()

#checks is_anat() function true cases
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

#checks is_anat() function false cases
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

#checks is_func_bold() true cases
def test_is_func_bold_true(testvalidator):
    target_list = ["/sub-01/func/sub-01_task-coding_bold.nii.gz",
    "/sub-01/func/sub-01_task-coding_sbref.nii.gz",
    "/sub-01/func/sub-01_task-coding_acq-23_bold.nii.gz",
    "/sub-01/func/sub-01_task-coding_acq-23_sbref.nii.gz",
    "/sub-01/func/sub-01_task-coding_run-23_bold.nii.gz",
    "/sub-01/func/sub-01_task-coding_run-23_sbref.nii.gz",
    "/sub-01/func/sub-01_task-coding_acq-23_run-23_bold.nii.gz",
    "/sub-01/func/sub-01_task-coding_acq-23_run-23_sbref.nii.gz",
    "/sub-01/ses-test/func/sub-01_ses-test_task-coding_bold.nii.gz",
    "/sub-01/ses-test/func/sub-01_ses-test_task-coding_sbref.nii.gz",
    "/sub-01/ses-test/func/sub-01_ses-test_task-coding_acq-23_bold.nii.gz",
    "/sub-01/ses-test/func/sub-01_ses-test_task-coding_acq-23_sbref.nii.gz",
    "/sub-01/ses-test/func/sub-01_ses-test_task-coding_run-23_bold.nii.gz",
    "/sub-01/ses-test/func/sub-01_ses-test_task-coding_run-23_sbref.nii.gz",
    "/sub-01/ses-test/func/sub-01_ses-test_task-coding_acq-23_run-23_bold.nii.gz",
    "/sub-01/ses-test/func/sub-01_ses-test_task-coding_acq-23_run-23_sbref.nii.gz"
    ]

    for item in target_list:
        result = testvalidator.is_func_bold(item)
        if result != True:
            print(item)
        assert result == True

#checks is_func_bold() false cases
def test_is_func_bold_true(testvalidator):
    target_list = [
#func not bold
"/sub-01/ses-test/func/sub-01_ses-test_task-coding_acq-23_events.tsv",
"/sub-01/func/sub-01_task-coding_events.json",
"/sub-01/func/sub-01_task-coding_acq-23_run-23_events.json",
"/sub-01/ses-test/func/sub-01_ses-test_task-coding_bold.json",
"/sub-01/func/sub-01_task-coding_acq-23_run-23_bold.json",
"/sub-01/ses-test/func/sub-01_ses-test_task-coding_physio.json",
"/sub-01/ses-test/func/sub-01_ses-test_task-coding_acq-23_run-23_physio.json",
"/sub-01/func/sub-01_task-coding_run-23_events.tsv",
"/sub-01/func/sub-01_task-coding_events.tsv",
"/sub-01/func/sub-01_task-coding_acq-23_events.json",
"/sub-01/ses-test/func/sub-01_ses-test_task-coding_run-23_events.tsv",
"/sub-01/func/sub-01_task-coding_physio.json",
"/sub-01/func/sub-01_task-coding_acq-23_physio.json",
# various typos
"/sub-01/func/sub-01_task-coding_sbref.ni.gz", # ni
"/sub-01/func/sub-01_task-coding_acq_23_bold.nii.gz", #_23
"/sub-01/func/sub-01_task-coding_acq-23_sbrf.nii.gz", #sbrf
"/sub-01/func/sub-02_task-coding_run-23_bold.nii.gz", #sub-02
"/sub-01/func/sub-01_task-coding-run-23_sbref.nii.gz", #-run
"/sub-01/func/sub-01_task_coding_acq-23_run-23_bold.nii.gz", #_coding
"/sub-01/func/sub-01-task-coding_acq-23_run-23_sbref.nii.gz", #-task
"/sub-01/ses-test/func/sub-01_ses-retest_task-coding_bold.nii.gz", #ses-retest
"/sub-01/ses-test/func/sub-02_ses-test_task-coding_sbref.nii.gz", #sub-02
"/sub-01/ses-test/func/sub-01_ses-test_task-coding_acq-23_blad.nii.gz", #blad
"/sub-01/ses-test/func/sub-01_ses-test-task-coding_acq-23_sbref.nii.gz", #-task
"/sub-01/ses-test/anat/sub-01_ses-test_task-coding_run-23_bold.nii.gz", #anat
"/sub-01/ses-test/anat/sub-01_ses-test_task-coding_run-23_sbref.nii.gz", #anat
"/sub-01/ses-test/dwi/sub-01_ses-test_task-coding_acq-23_run-23_bold.nii.gz", #dwi
"/sub-01/ses-test/dwi/sub-01_ses-test_task-coding_acq-23_run-23_sbref.nii.gz" #dwi
    ]

    for item in target_list:
        result = testvalidator.is_func_bold(item)
        if result != False:
            print(item)
        assert result == False
