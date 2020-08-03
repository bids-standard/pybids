"""Tests of BIDSValidator functionality."""

from os.path import join, dirname, abspath

import pytest

from bids_validator import BIDSValidator
from bids.layout import BIDSLayout
from bids.tests import get_test_data_path


# Fixture uses in the rest of the tests
@pytest.fixture
def testvalidator():
    return BIDSValidator()


# checks is_top_level() function true cases
def test_is_top_level_true(testvalidator):
    target_list = [
        "/README",
        "/CHANGES",
        "/dataset_description.json",
        "/participants.tsv",
        "/participants.json"
    ]

    for item in target_list:
        result = testvalidator.is_top_level(item)
        assert result


# checks is_top_level() function false cases
def test_is_top_level_false(testvalidator):
    target_list = [
        "/RADME",  # wrong the the filename
        "/CANGES",  # wrong the the filename
        "/dataset_descrption.json",  # wrong the the filename
        "/dataset_description.jon",  # wrong extension
        "/participants.sv",  # wrong extension
        "/participnts.tsv",  # wrong the the filename
        "/particpants.json"  # wrong the the filename
        "/participants.son"  # wrong extension
    ]

    for item in target_list:
        result = testvalidator.is_top_level(item)
        assert not result


# checks is_associated_data() function true cases
def test_is_associated_data_true(testvalidator):
    target_list = [
        "/code/",
        "/derivatives/",
        "/sourcedata/",
        "/stimuli/",
    ]

    for item in target_list:
        result = testvalidator.is_associated_data(item)
        assert result


# checks is_associated_data() function false cases
def test_is_associated_data_false(testvalidator):
    target_list = [
        "/CODE/",
        "/derivatves/",
        "/source/",
        "/stimli/",
        "/.git/"
    ]

    for item in target_list:
        result = testvalidator.is_associated_data(item)
        assert not result


# checks is_session_level() function true cases
def test_is_session_level_true(testvalidator):
    target_list = [
        "/sub-01/sub-01_dwi.bval",
        "/sub-01/sub-01_dwi.bvec",
        "/sub-01/sub-01_dwi.json",
        "/sub-01/sub-01_run-01_dwi.bval",
        "/sub-01/sub-01_run-01_dwi.bvec",
        "/sub-01/sub-01_run-01_dwi.json",
        "/sub-01/sub-01_acq-singleband_dwi.bval",
        "/sub-01/sub-01_acq-singleband_dwi.bvec",
        "/sub-01/sub-01_acq-singleband_dwi.json",
        "/sub-01/sub-01_acq-singleband_run-01_dwi.bval",
        "/sub-01/sub-01_acq-singleband_run-01_dwi.bvec",
        "/sub-01/sub-01_acq-singleband_run-01_dwi.json",
        "/sub-01/ses-test/sub-01_ses-test_dwi.bval",
        "/sub-01/ses-test/sub-01_ses-test_dwi.bvec",
        "/sub-01/ses-test/sub-01_ses-test_dwi.json",
        "/sub-01/ses-test/sub-01_ses-test_run-01_dwi.bval",
        "/sub-01/ses-test/sub-01_ses-test_run-01_dwi.bvec",
        "/sub-01/ses-test/sub-01_ses-test_run-01_dwi.json",
        "/sub-01/ses-test/sub-01_ses-test_acq-singleband_dwi.bval",
        "/sub-01/ses-test/sub-01_ses-test_acq-singleband_dwi.bvec",
        "/sub-01/ses-test/sub-01_ses-test_acq-singleband_dwi.json",
        "/sub-01/ses-test/sub-01_ses-test_acq-singleband_run-01_dwi.bval",
        "/sub-01/ses-test/sub-01_ses-test_acq-singleband_run-01_dwi.bvec",
        "/sub-01/ses-test/sub-01_ses-test_acq-singleband_run-01_dwi.json"
    ]

    for item in target_list:
        result = testvalidator.is_session_level(item)
        assert result


# checks is_session_level() function false cases
def test_is_session_level_false(testvalidator):
    target_list = [
        "/sub-01/ses-ses/sub-01_dwi.bval",  # redundant dir /ses-ses/
        "/sub-01/01_dwi.bvec",  # missed subject suffix
        "/sub-01/sub_dwi.json",  # missed subject id
        "/sub-01/sub-01_23_run-01_dwi.bval",  # wrong _23_
        "/sub-01/sub-01_run-01_dwi.vec",  # wrong extension
        "/sub-01/sub-01_run-01_dwi.jsn",  # wrong extension
        "/sub-01/sub-01_acq_dwi.bval",  # missed suffix value
        "/sub-01/sub-01_acq-23-singleband_dwi.bvec",  # redundant -23-
        "/sub-01/anat/sub-01_acq-singleband_dwi.json",  # redundant /anat/
        "/sub-01/sub-01_recrod-record_acq-singleband_run-01_dwi.bval", # redundant record-record_
        "/sub_01/sub-01_acq-singleband_run-01_dwi.bvec",  # wrong /sub_01/
        "/sub-01/sub-01_acq-singleband__run-01_dwi.json",  # wrong __
        "/sub-01/ses-test/sub-01_ses_test_dwi.bval",  # wrong ses_test
        "/sub-01/ses-test/sb-01_ses-test_dwi.bvec",  # wrong sb-01
        "/sub-01/ses-test/sub-01_ses-test_dw.json",  # wrong modality
        "/sub-01/ses-test/sub-01_ses-test_run-01_dwi.val",  # wrong extension
        "/sub-01/ses-test/sub-01_run-01_dwi.bvec",  # missed session in the filename
        "/sub-01/ses-test/ses-test_run-01_dwi.json",  # missed subject in the filename
        "/sub-01/ses-test/sub-01_ses-test_acq-singleband.bval",  # missed modality
        "/sub-01/ses-test/sub-01_ses-test_acq-singleband_dwi",  # missed extension
        "/ses-test/sub-01/sub-01_ses-test_acq-singleband_dwi.json",  # wrong dirs order
        "/sub-01/ses-test/sub-02_ses-test_acq-singleband_run-01_dwi.bval", # wrong sub id in the filename
        "/sub-01/sub-01_ses-test_acq-singleband_run-01_dwi.bvec",  # ses dir missed
        "/ses-test/sub-01_ses-test_acq-singleband_run-01_dwi.json"  # sub id dir missed
    ]

    for item in target_list:
        result = testvalidator.is_session_level(item)
        assert not result


# checks is_subject_level() function true cases
def test_is_subject_level_true(testvalidator):
    target_list = [
        "/sub-01/sub-01_sessions.tsv",
        "/sub-01/sub-01_sessions.json"
    ]

    for item in target_list:
        result = testvalidator.is_subject_level(item)
        assert result


# checks is_subject_level() function false cases
def test_is_subject_false(testvalidator):
    target_list = [
        "/sub-02/sub-01_sessions.tsv",  # wrong sub id in the filename
        "/sub-01_sessions.tsv",  # missed subject id dir
        "/sub-01/sub-01_sesions.tsv",  # wrong modality
        "/sub-01/sub-01_sesions.ext",  # wrong extension
        "/sub-01/sub-01_sessions.jon"  # wrong extension
    ]

    for item in target_list:
        result = testvalidator.is_subject_level(item)
        assert not result


# # checks is_anat() function true cases
# def test_is_anat_true(testvalidator):
#     target_list = [
#         "/sub-01/anat/sub-01_T1w.json",
#         "/sub-01/anat/sub-01_T1w.nii.gz",
#         "/sub-01/anat/sub-01_rec-CSD_T1w.json",
#         "/sub-01/anat/sub-01_rec-CSD_T1w.nii.gz",
#         "/sub-01/anat/sub-01_acq-23_T1w.json",
#         "/sub-01/anat/sub-01_acq-23_T1w.nii.gz",
#         "/sub-01/anat/sub-01_acq-23_rec-CSD_T1w.json",
#         "/sub-01/anat/sub-01_acq-23_rec-CSD_T1w.nii.gz",
#         "/sub-01/anat/sub-01_run-23_T1w.json",
#         "/sub-01/anat/sub-01_run-23_T1w.nii.gz",
#         "/sub-01/anat/sub-01_rec-CSD_run-23_T1w.json",
#         "/sub-01/anat/sub-01_rec-CSD_run-23_T1w.nii.gz",
#         "/sub-01/anat/sub-01_acq-23_run-23_T1w.json",
#         "/sub-01/anat/sub-01_acq-23_run-23_T1w.nii.gz",
#         "/sub-01/anat/sub-01_acq-23_rec-CSD_run-23_T1w.json",
#         "/sub-01/anat/sub-01_acq-23_rec-CSD_run-23_T1w.nii.gz",
#         "/sub-01/ses-test/anat/sub-01_ses-test_T1w.json",
#         "/sub-01/ses-test/anat/sub-01_ses-test_T1w.nii.gz",
#         "/sub-01/ses-test/anat/sub-01_ses-test_rec-CSD_T1w.json",
#         "/sub-01/ses-test/anat/sub-01_ses-test_rec-CSD_T1w.nii.gz",
#         "/sub-01/ses-test/anat/sub-01_ses-test_acq-23_T1w.json",
#         "/sub-01/ses-test/anat/sub-01_ses-test_acq-23_T1w.nii.gz",
#         "/sub-01/ses-test/anat/sub-01_ses-test_acq-23_rec-CSD_T1w.json",
#         "/sub-01/ses-test/anat/sub-01_ses-test_acq-23_rec-CSD_T1w.nii.gz",
#         "/sub-01/ses-test/anat/sub-01_ses-test_run-23_T1w.json",
#         "/sub-01/ses-test/anat/sub-01_ses-test_run-23_T1w.nii.gz",
#         "/sub-01/ses-test/anat/sub-01_ses-test_rec-CSD_run-23_T1w.json",
#         "/sub-01/ses-test/anat/sub-01_ses-test_rec-CSD_run-23_T1w.nii.gz",
#         "/sub-01/ses-test/anat/sub-01_ses-test_acq-23_run-23_T1w.json",
#         "/sub-01/ses-test/anat/sub-01_ses-test_acq-23_run-23_T1w.nii.gz",
#         "/sub-01/ses-test/anat/sub-01_ses-test_acq-23_rec-CSD_run-23_T1w.json",
#         "/sub-01/ses-test/anat/sub-01_ses-test_acq-23_rec-CSD_run-23_T1w.nii.gz"]

#     for item in target_list:
#         result = testvalidator.is_anat(item)
#         assert result

# # checks is_anat() function false cases


# def test_is_anat_false(testvalidator):
#     target_list = ["/sub-01/anat/sub-1_T1w.json",  # subject incosistency
#                    "/sub-01/anat/sub-01_dwi.nii.gz",  # wrong modality suffix
#                    "/sub-01/anat/sub-02_rec-CSD_T1w.json",  # subject incosistency
#                    "/sub-01/anat/sub-01_rec-CS-D_T1w.nii.gz",  # rec label wrong
#                    "/sub-01/anat/sub-01_acq-23_T1W.json",  # modality suffix wrong
#                    "/sub-01/dwi/sub-01_acq-23_dwi.nii.gz",  # wrong data type
#                    "/sub-01/anat/sub-01_acq-23_rec-CSD_T1w.exe",  # wrong extension
#                    "/sub-01/anat/sub-01_acq-23_rec-CSD_T1w.niigz",  # extension typo
#                    "/sub-01/anat/sub-01_run-2-3_T1w.json",  # run label typo
#                    "/sub-01/anat/sub-01_rn-23_T1w.nii.gz",  # run typo
#                    "/sub-01/ant/sub-01_rec-CS-D_run-23_T1w.json",  # reconstruction label typo
#                    "/sub-1/anat/sub-01_rec-CSD_run-23_t1w.nii.gz",  # T1w suffix typo
#                    "/sub-01/anat/sub-01_aq-23_run-23_T1w.json",  # acq typo
#                    "/sub-01/anat/sub-01_acq-23_run-23_dwi.nii.gz",  # wrong data type
#                    "/sub-01/anat/sub-01_acq-23_rc-CSD_run-23_T1w.json",  # rec typo
#                    "/sub-01/anat/sub-O1_acq-23_rec-CSD_run-23_T1w.nii.gz",  # 2nd subject id typo
#                    "/sub-01/ses-test/anat/sub-01_ses-retest_T1w.json",  # ses inconsistency
#                    "/sub-01/ses-test/anat/sub-01_sestest_T1w.nii.gz",  # 2nd session typo
#                    "/sub-01/ses-test/anat/sub-01_ses_test_rec-CSD_dwi.jsn",  # extension typo
#                    "/sub-01/ses_test/anat/sub-01_ses_test_rec-CSD_T1w.bval",  # wrong extension
#                    "/sub-01/ses-test/anat/sub-01_ses-test_acq-23_T1w.exe",  # wrong extension
#                    ]
#     for item in target_list:
#         result = testvalidator.is_anat(item)
#         assert not result


# # checks is_dwi() function true cases
# def test_is_dwi_true(testvalidator):
#     target_list = [
#         "/sub-01/dwi/sub-01_dwi.nii.gz",
#         "/sub-01/dwi/sub-01_dwi.bval",
#         "/sub-01/dwi/sub-01_dwi.bvec",
#         "/sub-01/dwi/sub-01_dwi.json",
#         "/sub-01/dwi/sub-01_run-01_dwi.nii.gz",
#         "/sub-01/dwi/sub-01_run-01_dwi.bval",
#         "/sub-01/dwi/sub-01_run-01_dwi.bvec",
#         "/sub-01/dwi/sub-01_run-01_dwi.json",
#         "/sub-01/dwi/sub-01_acq-singleband_dwi.nii.gz",
#         "/sub-01/dwi/sub-01_acq-singleband_dwi.bval",
#         "/sub-01/dwi/sub-01_acq-singleband_dwi.bvec",
#         "/sub-01/dwi/sub-01_acq-singleband_dwi.json",
#         "/sub-01/dwi/sub-01_acq-singleband_run-01_dwi.nii.gz",
#         "/sub-01/dwi/sub-01_acq-singleband_run-01_dwi.bval",
#         "/sub-01/dwi/sub-01_acq-singleband_run-01_dwi.bvec",
#         "/sub-01/dwi/sub-01_acq-singleband_run-01_dwi.json",
#         "/sub-01/ses-test/dwi/sub-01_ses-test_dwi.nii.gz",
#         "/sub-01/ses-test/dwi/sub-01_ses-test_dwi.bval",
#         "/sub-01/ses-test/dwi/sub-01_ses-test_dwi.bvec",
#         "/sub-01/ses-test/dwi/sub-01_ses-test_dwi.json",
#         "/sub-01/ses-test/dwi/sub-01_ses-test_run-01_dwi.nii.gz",
#         "/sub-01/ses-test/dwi/sub-01_ses-test_run-01_dwi.bval",
#         "/sub-01/ses-test/dwi/sub-01_ses-test_run-01_dwi.bvec",
#         "/sub-01/ses-test/dwi/sub-01_ses-test_run-01_dwi.json",
#         "/sub-01/ses-test/dwi/sub-01_ses-test_acq-singleband_dwi.nii.gz",
#         "/sub-01/ses-test/dwi/sub-01_ses-test_acq-singleband_dwi.bval",
#         "/sub-01/ses-test/dwi/sub-01_ses-test_acq-singleband_dwi.bvec",
#         "/sub-01/ses-test/dwi/sub-01_ses-test_acq-singleband_dwi.json",
#         "/sub-01/ses-test/dwi/sub-01_ses-test_acq-singleband_run-01_dwi.nii.gz",
#         "/sub-01/ses-test/dwi/sub-01_ses-test_acq-singleband_run-01_dwi.bval",
#         "/sub-01/ses-test/dwi/sub-01_ses-test_acq-singleband_run-01_dwi.bvec",
#         "/sub-01/ses-test/dwi/sub-01_ses-test_acq-singleband_run-01_dwi.json"]

#     for item in target_list:
#         result = testvalidator.is_dwi(item)
#         assert result

# # checks is_dwi() function false cases


# def test_is_dwi_false(testvalidator):
#     target_list = [
#         "/sub-01/dwi/sub-01_suffix-suff_acq-singleband_dwi.json",  # redundant suffix
#         "/sub-01/dwi/sub-01_acq-singleband__run-01_dwi.nii.gz",  # wrong __
#         "/sub-01/dwi/sub-01_acq_run-01_dwi.bval",  # missed -singleband in _acq
#         "/sub-01/dwi/sub-01_acq-singleband_run_01_dwi.bvec",  # wrong run_01
#         "/sub-01/dwi/sub-01_acq_singleband_run-01_dwi.json",  # wrong acq_singleband_
#         "/sub_01/ses-test/dwi/sub-01_ses-test_dwi.nii.gz",  # wrong sub_01 dir
#         "/sub-01/ses_test/dwi/sub-01_ses-test_dwi.bval",  # wrong ses_test dir
#         "/sub-01/ses-retest/dwi/sub-01_ses-test_dwi.bvec",  # wrong session in the filename
#         "/sub-01/ses-test/dwi/sub-01_ses-retest_dwi.json",  # wrong session in the filename
#         "/sub-01/ses-test/dwi/sub-01_ses-test_run-01_brain.nii.gz",  # wrong modality
#         "/sub-01/ses-test/dwi/sub-01_ses-test_run-01.bval",  # missed modality
#         "/sub-01/ses-test/dwi/sub-01_ses-test_run-01_dwi.vec",  # wrong extension
#         "/sub-01/ses-test/dwi/sub-01_ses-test_run-01_dwi.jon",  # wrong extension
#         "/sub-01/ses-test/dwi/sub-01_ses-test_acq-singleband_dwi.ni.gz",  # wrong extension
#         "/sub-01/ses-test/dwi/sub-01_ses-test_acq-singleband_dwi.val",  # wrong extension
#         "/ses-test/dwi/sub-01/sub-01_ses-test_acq-singleband_dwi.bvec",  # wrong dirs order
#         "/sub-01/dwi/ses-test/sub-01_ses-test_acq-singleband_dwi.json",  # wrong dirs order
#         "/ses-test/sub-01/dwi/sub-01_ses-test_acq-singleband_run-01_dwi.nii.gz",  # wrong dirs order
#         "/sub-01/ses-test/sub-01_ses-test_acq-singleband_run-01_dwi.bval",  # missed data type dir
#         "/sub-01/dwi/sub-01_ses-test_acq-singleband_run-01_dwi.bvec",  # missed session id dir
#         "/ses-test/dwi/sub-01_ses-test_acq-singleband_run-01_dwi.json"  # missed sub id dir
#     ]

#     for item in target_list:
#         result = testvalidator.is_dwi(item)
#         assert not result


# # checks is_func() function true cases
# def test_is_func_true(testvalidator):
#     target_list = [
#         "/sub-01/func/sub-01_task-task_bold.nii.gz",
#         "/sub-01/func/sub-01_task-task_bold.nii",
#         "/sub-01/func/sub-01_task-task_bold.json",
#         "/sub-01/func/sub-01_task-task_sbref.nii.gz",
#         "/sub-01/func/sub-01_task-task_sbref.json",
#         "/sub-01/func/sub-01_task-task_events.json",
#         "/sub-01/func/sub-01_task-task_events.tsv",
#         "/sub-01/func/sub-01_task-task_physio.json",
#         "/sub-01/func/sub-01_task-task_physio.tsv.gz",
#         "/sub-01/func/sub-01_task-task_stim.json",
#         "/sub-01/func/sub-01_task-task_stim.tsv.gz",
#         "/sub-01/func/sub-01_task-task_defacemask.nii.gz",
#         "/sub-01/func/sub-01_task-task_defacemask.nii",
#         "/sub-01/func/sub-01_task-task_run-01_bold.nii.gz",
#         "/sub-01/func/sub-01_task-task_run-01_bold.nii",
#         "/sub-01/func/sub-01_task-task_run-01_bold.json",
#         "/sub-01/func/sub-01_task-task_run-01_sbref.nii.gz",
#         "/sub-01/func/sub-01_task-task_run-01_sbref.json",
#         "/sub-01/func/sub-01_task-task_run-01_events.json",
#         "/sub-01/func/sub-01_task-task_run-01_events.tsv",
#         "/sub-01/func/sub-01_task-task_run-01_physio.json",
#         "/sub-01/func/sub-01_task-task_run-01_physio.tsv.gz",
#         "/sub-01/func/sub-01_task-task_run-01_stim.json",
#         "/sub-01/func/sub-01_task-task_run-01_stim.tsv.gz",
#         "/sub-01/func/sub-01_task-task_run-01_defacemask.nii.gz",
#         "/sub-01/func/sub-01_task-task_run-01_defacemask.nii",
#         "/sub-01/func/sub-01_task-task_rec-rec_bold.nii.gz",
#         "/sub-01/func/sub-01_task-task_rec-rec_bold.nii",
#         "/sub-01/func/sub-01_task-task_rec-rec_bold.json",
#         "/sub-01/func/sub-01_task-task_rec-rec_sbref.nii.gz",
#         "/sub-01/func/sub-01_task-task_rec-rec_sbref.json",
#         "/sub-01/func/sub-01_task-task_rec-rec_events.json",
#         "/sub-01/func/sub-01_task-task_rec-rec_events.tsv",
#         "/sub-01/func/sub-01_task-task_rec-rec_physio.json",
#         "/sub-01/func/sub-01_task-task_rec-rec_physio.tsv.gz",
#         "/sub-01/func/sub-01_task-task_rec-rec_stim.json",
#         "/sub-01/func/sub-01_task-task_rec-rec_stim.tsv.gz",
#         "/sub-01/func/sub-01_task-task_rec-rec_defacemask.nii.gz",
#         "/sub-01/func/sub-01_task-task_rec-rec_defacemask.nii",
#         "/sub-01/func/sub-01_task-task_rec-rec_run-01_bold.nii.gz",
#         "/sub-01/func/sub-01_task-task_rec-rec_run-01_bold.nii",
#         "/sub-01/func/sub-01_task-task_rec-rec_run-01_bold.json",
#         "/sub-01/func/sub-01_task-task_rec-rec_run-01_sbref.nii.gz",
#         "/sub-01/func/sub-01_task-task_rec-rec_run-01_sbref.json",
#         "/sub-01/func/sub-01_task-task_rec-rec_run-01_events.json",
#         "/sub-01/func/sub-01_task-task_rec-rec_run-01_events.tsv",
#         "/sub-01/func/sub-01_task-task_rec-rec_run-01_physio.json",
#         "/sub-01/func/sub-01_task-task_rec-rec_run-01_physio.tsv.gz",
#         "/sub-01/func/sub-01_task-task_rec-rec_run-01_stim.json",
#         "/sub-01/func/sub-01_task-task_rec-rec_run-01_stim.tsv.gz",
#         "/sub-01/func/sub-01_task-task_rec-rec_run-01_defacemask.nii.gz",
#         "/sub-01/func/sub-01_task-task_rec-rec_run-01_defacemask.nii",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_bold.nii.gz",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_bold.nii",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_bold.json",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_sbref.nii.gz",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_sbref.json",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_events.json",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_events.tsv",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_physio.json",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_physio.tsv.gz",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_stim.json",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_stim.tsv.gz",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_defacemask.nii.gz",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_defacemask.nii",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_run-01_bold.nii.gz",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_run-01_bold.nii",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_run-01_bold.json",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_run-01_sbref.nii.gz",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_run-01_sbref.json",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_run-01_events.json",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_run-01_events.tsv",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_run-01_physio.json",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_run-01_physio.tsv.gz",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_run-01_stim.json",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_run-01_stim.tsv.gz",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_run-01_defacemask.nii.gz",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_run-01_defacemask.nii",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_rec-rec_bold.nii.gz",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_rec-rec_bold.nii",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_rec-rec_bold.json",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_rec-rec_sbref.nii.gz",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_rec-rec_sbref.json",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_rec-rec_events.json",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_rec-rec_events.tsv",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_rec-rec_physio.json",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_rec-rec_physio.tsv.gz",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_rec-rec_stim.json",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_rec-rec_stim.tsv.gz",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_rec-rec_defacemask.nii.gz",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_rec-rec_defacemask.nii",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_rec-rec_run-01_bold.nii.gz",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_rec-rec_run-01_bold.nii",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_rec-rec_run-01_bold.json",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_rec-rec_run-01_sbref.nii.gz",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_rec-rec_run-01_sbref.json",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_rec-rec_run-01_events.json",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_rec-rec_run-01_events.tsv",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_rec-rec_run-01_physio.json",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_rec-rec_run-01_physio.tsv.gz",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_rec-rec_run-01_stim.json",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_rec-rec_run-01_stim.tsv.gz",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_rec-rec_run-01_defacemask.nii.gz",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_rec-rec_run-01_defacemask.nii"]

#     for item in target_list:
#         result = testvalidator.is_func(item)
#         assert result


# # checks is_func() function false cases
# def test_is_func_false(testvalidator):
#     target_list = [
#         "/sub-01/ses-test/func/sub--01_ses-test_task-task_rec-rec_stim.tsv.gz",  # wrong --
#         "/sub-01/ses-test/func/sub-01__ses-test_task-task_rec-rec_defacemask.nii.gz",  # wrong __
#         "/sub-01/ses-test/func/sub-01_ses_test_task-task_rec-rec_defacemask.nii",  # wrong ses_test
#         "/sub-01/ses-test/func/sub-01_task-task_rec-rec_run-01_bold.nii.gz", # missed session suffix and id in the filename
#         "/sub-01/ses-test/func/ses-test_task-task_rec-rec_run-01_bold.nii", # missed subject suffix and id in the filename
#         "/sub-01/ses-retest/func/sub-01_ses-test_task-task_rec-rec_run-01_sbref.nii.gz", # wrong session id in teh filename
#         "/sub-01/ses-test/func/sub-02_ses-test_task-task_rec-rec_run-01_sbref.json", # wrong subject id in the filename
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_rec-rec_run-01.json",  # missed modality
#         "/sub-01/ses-test/func/sub-01_ses-test_task-task_rec-rec_run-01_events",  # missed extension
#         "/sub-01/func/ses-test/sub-01_ses-test_task-task_rec-rec_run-01_physio.json", # wrong dirs order
#         "/ses-test/func/sub-01/sub-01_ses-test_task-task_rec-rec_run-01_physio.tsv.gz", # wrong dirs order
#         "/ses-test/sub-01/func/sub-01_ses-test_task-task_rec-rec_run-01_stim.json", # wrong dirs order
#         "/sub-01/ses-test/sub-01_ses-test_task-task_rec-rec_run-01_stim.tsv.gz",  # missed data type
#         "/sub-01/func/sub-01_ses-test_task-task_rec-rec_run-01_defacemask.nii.gz", # missed session dir
#         "/ses-test/func/sub-01_ses-test_task-task_rec-rec_run-01_defacemask.nii" # missed subject dir
#     ]

#     for item in target_list:
#         result = testvalidator.is_func(item)
#         assert not result


# # checks is_func_bold() true cases
# def test_is_func_bold_true(testvalidator):
#     target_list = [
#         "/sub-01/func/sub-01_task-coding_bold.nii.gz",
#         "/sub-01/func/sub-01_task-coding_sbref.nii.gz",
#         "/sub-01/func/sub-01_task-coding_acq-23_bold.nii.gz",
#         "/sub-01/func/sub-01_task-coding_acq-23_sbref.nii.gz",
#         "/sub-01/func/sub-01_task-coding_run-23_bold.nii.gz",
#         "/sub-01/func/sub-01_task-coding_run-23_sbref.nii.gz",
#         "/sub-01/func/sub-01_task-coding_acq-23_run-23_bold.nii.gz",
#         "/sub-01/func/sub-01_task-coding_acq-23_run-23_sbref.nii.gz",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-coding_bold.nii.gz",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-coding_sbref.nii.gz",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-coding_acq-23_bold.nii.gz",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-coding_acq-23_sbref.nii.gz",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-coding_run-23_bold.nii.gz",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-coding_run-23_sbref.nii.gz",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-coding_acq-23_run-23_bold.nii.gz",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-coding_acq-23_run-23_sbref.nii.gz"]

#     for item in target_list:
#         result = testvalidator.is_func_bold(item)
#         assert result


# # checks is_func_bold() false cases
# def test_is_func_bold_false(testvalidator):
#     target_list = [
#         # func not bold
#         "/sub-01/ses-test/func/sub-01_ses-test_task-coding_acq-23_events.tsv",
#         "/sub-01/func/sub-01_task-coding_events.json",
#         "/sub-01/func/sub-01_task-coding_acq-23_run-23_events.json",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-coding_bold.json",
#         "/sub-01/func/sub-01_task-coding_acq-23_run-23_bold.json",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-coding_physio.json",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-coding_acq-23_run-23_physio.json",
#         "/sub-01/func/sub-01_task-coding_run-23_events.tsv",
#         "/sub-01/func/sub-01_task-coding_events.tsv",
#         "/sub-01/func/sub-01_task-coding_acq-23_events.json",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-coding_run-23_events.tsv",
#         "/sub-01/func/sub-01_task-coding_physio.json",
#         "/sub-01/func/sub-01_task-coding_acq-23_physio.json",
#         # various typos
#         "/sub-01/func/sub-01_task-coding_sbref.ni.gz",  # ni
#         "/sub-01/func/sub-01_task-coding_acq_23_bold.nii.gz",  # _23
#         "/sub-01/func/sub-01_task-coding_acq-23_sbrf.nii.gz",  # sbrf
#         "/sub-01/func/sub-02_task-coding_run-23_bold.nii.gz",  # sub-02
#         "/sub-01/func/sub-01_task-coding-run-23_sbref.nii.gz",  # -run
#         "/sub-01/func/sub-01_task_coding_acq-23_run-23_bold.nii.gz",  # _coding
#         "/sub-01/func/sub-01-task-coding_acq-23_run-23_sbref.nii.gz",  # -task
#         "/sub-01/ses-test/func/sub-01_ses-retest_task-coding_bold.nii.gz",  # ses-retest
#         "/sub-01/ses-test/func/sub-02_ses-test_task-coding_sbref.nii.gz",  # sub-02
#         "/sub-01/ses-test/func/sub-01_ses-test_task-coding_acq-23_blad.nii.gz",  # blad
#         "/sub-01/ses-test/func/sub-01_ses-test-task-coding_acq-23_sbref.nii.gz",  # -task
#         "/sub-01/ses-test/anat/sub-01_ses-test_task-coding_run-23_bold.nii.gz",  # anat
#         "/sub-01/ses-test/anat/sub-01_ses-test_task-coding_run-23_sbref.nii.gz",  # anat
#         "/sub-01/ses-test/dwi/sub-01_ses-test_task-coding_acq-23_run-23_bold.nii.gz",  # dwi
#         "/sub-01/ses-test/dwi/sub-01_ses-test_task-coding_acq-23_run-23_sbref.nii.gz"  # dwi
#     ]

#     for item in target_list:
#         result = testvalidator.is_func_bold(item)
#         assert not result


# # checks is_behavioral() function true cases
# def test_is_behavioral_true(testvalidator):
#     target_list = [
#         "/sub-01/beh/sub-01_task-task_events.tsv",
#         "/sub-01/beh/sub-01_task-task_events.json",
#         "/sub-01/beh/sub-01_task-task_beh.json",
#         "/sub-01/beh/sub-01_task-task_physio.json",
#         "/sub-01/beh/sub-01_task-task_physio.tsv.gz",
#         "/sub-01/beh/sub-01_task-task_stim.json",
#         "/sub-01/beh/sub-01_task-task_stim.tsv.gz",
#         "/sub-01/ses-test/beh/sub-01_ses-test_task-task_events.tsv",
#         "/sub-01/ses-test/beh/sub-01_ses-test_task-task_events.json",
#         "/sub-01/ses-test/beh/sub-01_ses-test_task-task_beh.json",
#         "/sub-01/ses-test/beh/sub-01_ses-test_task-task_physio.json",
#         "/sub-01/ses-test/beh/sub-01_ses-test_task-task_physio.tsv.gz",
#         "/sub-01/ses-test/beh/sub-01_ses-test_task-task_stim.json",
#         "/sub-01/ses-test/beh/sub-01_ses-test_task-task_stim.tsv.gz",
#     ]

#     for item in target_list:
#         result = testvalidator.is_behavioral(item)
#         assert result


# # checks is_behavioral() function false cases
# def test_is_behavioral_false(testvalidator):
#     target_list = [
#         "/sub-01/beeh/sub-01_task-task_events.tsv",  # wrong data type
#         "/sub-01/beh/sub-01_suff-suff_task-task_events.json",  # wrong suffix
#         "/sub-01/beh/sub-02_task-task_beh.json",  # wrong sub id in the filename
#         "/sub-01/beh/sub-01_task_task_physio.json",  # wrong task_task
#         "/sub-01/beh/sub-01_task-task_phycoo.tsv.gz",  # wrong modality
#         "/sub-01/beh/sub-01_task-task_stim.jsn",  # wrong extension
#         "/sub-01/beh/sub-01_task-task.tsv.gz",  # missed modality
#         "/sub-01/ses-test/beh/sub-01_ses-test_task-task_events",  # missed extension
#         "/sub-01/beh/ses-test/sub-01_ses-test_task-task_events.json",  # wrong dirs order
#         "/ses-test/beh/sub-01/sub-01_ses-test_task-task_beh.json",  # wrong dirs order
#         "/ses-test/sub-01/beh/sub-01_ses-test_task-task_physio.json",  # wrong dirs order
#         "/sub-01/ses-test/sub-01_ses-test_task-task_physio.tsv.gz",  # missed data type dir
#         "/sub-01/beh/sub-01_ses-test_task-task_stim.json",  # missed session id dir
#         "/ses-test/beh/sub-01_ses-test_task-task_stim.tsv.gz",  # missed subject id dir
#     ]

#     for item in target_list:
#         result = testvalidator.is_behavioral(item)
#         assert not result


# # checks is_cont() function true cases
# def test_is_cont_true(testvalidator):
#     target_list = [
#         "/sub-01/ses-test/func/sub-01_ses-test_task-nback_physio.tsv.gz",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-nback_physio.json",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-nback_stim.tsv.gz",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-nback_stim.json",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-nback_recording-saturation_physio.tsv.gz",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-nback_recording-saturation_physio.json",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-nback_recording-saturation_stim.tsv.gz",
#         "/sub-01/ses-test/func/sub-01_ses-test_task-nback_recording-saturation_stim.json",
#         "/sub-01/ses-test/beh/sub-01_ses-test_task-nback_physio.tsv.gz",
#         "/sub-01/ses-test/beh/sub-01_ses-test_task-nback_physio.json",
#         "/sub-01/ses-test/beh/sub-01_ses-test_task-nback_stim.tsv.gz",
#         "/sub-01/ses-test/beh/sub-01_ses-test_task-nback_stim.json",
#         "/sub-01/ses-test/beh/sub-01_ses-test_task-nback_recording-saturation_physio.tsv.gz",
#         "/sub-01/ses-test/beh/sub-01_ses-test_task-nback_recording-saturation_physio.json",
#         "/sub-01/ses-test/beh/sub-01_ses-test_task-nback_recording-saturation_stim.tsv.gz",
#         "/sub-01/ses-test/beh/sub-01_ses-test_task-nback_recording-saturation_stim.json",
#     ]

#     for item in target_list:
#         result = testvalidator.is_cont(item)
#         assert result


# # checks is_cont() function false cases
# def test_is_cont_false(testvalidator):
#     target_list = [
#         "/sub-01/ses-test/func/sub--01_ses-test_task-nback_physio.tsv.gz",  # wrong --
#         "/sub-01/ses-test/func/sub-01__ses-test_task-nback_physio.json",  # wrong __
#         "/sb-01/ses-test/func/sub-01_ses-test_task-nback_stim.tsv.gz",  # wrong subject dir
#         "/sub-01/ss-test/func/sub-01_ses-test_task-nback_stim.json",  # wrong  session dir
#         "/sub-01/ses-test/dwi/sub-01_ses-test_task-nback_recording-saturation_physio.tsv.gz", # wrong data type
#         "/sub-01/ses-test/func/sub-01_ses-test_tsk-nback_recording-saturation_physio.json", # wrong suffix tsk-
#         "/sub-01/ses-test/func/sub-01_ses-retest_task-nback_recording-saturation_stim.tsv.gz", # wrong session id in the filename
#         "/sub_01/ses-test/func/sub-02_ses-test_task-nback_recording-saturation_stim.json", # wrong subject id in the filename
#         "/sub-01/beh/ses-test/sub-01_ses-test_task-nback_physio.tsv.gz",  # wrong dirs order
#         "/ses-test/beh/sub-01/sub-01_ses-test_task-nback_physio.json",  # wrong dirs order
#         "/ses-test/sub-01/beh/sub-01_ses-test_task-nback_stim.tsv.gz",  # wrong dirs order
#         "/sub-01/ses-test/beh/sub-01_ses-test_task-nback.json",  # missed modality
#         "/sub-01/ses-test/beh/sub-01_ses-test_task-nback_recording-saturation_physio.", # missed extension
#         "/sub-01/ses-test/sub-01_ses-test_task-nback_recording-saturation_physio.json", # missed data type dir
#         "/sub-01/beh/sub-01_ses-test_task-nback_recording-saturation_stim.tsv.gz", # missed session id dir
#         "/ses-test/beh/sub-01_ses-test_task-nback_recording-saturation_stim.json" # missed sub id dir
#     ]

#     for item in target_list:
#         result = testvalidator.is_cont(item)
#         assert not result


# # checks is_field_map() function true cases
# def test_is_field_map_true(testvalidator):
#     target_list = [
#         "/sub-01/fmap/sub-01_phasediff.nii.gz",
#         "/sub-01/fmap/sub-01_phasediff.json",
#         "/sub-01/fmap/sub-01_phasediff.nii",
#         "/sub-01/fmap/sub-01_phase1.nii.gz",
#         "/sub-01/fmap/sub-01_phase1.json",
#         "/sub-01/fmap/sub-01_phase1.nii",
#         "/sub-01/fmap/sub-01_phase2.nii.gz",
#         "/sub-01/fmap/sub-01_phase2.json",
#         "/sub-01/fmap/sub-01_phase2.nii",
#         "/sub-01/fmap/sub-01_magnitude.nii.gz",
#         "/sub-01/fmap/sub-01_magnitude.json",
#         "/sub-01/fmap/sub-01_magnitude.nii",
#         "/sub-01/fmap/sub-01_magnitude1.nii.gz",
#         "/sub-01/fmap/sub-01_magnitude1.json",
#         "/sub-01/fmap/sub-01_magnitude1.nii",
#         "/sub-01/fmap/sub-01_magnitude2.nii.gz",
#         "/sub-01/fmap/sub-01_magnitude2.json",
#         "/sub-01/fmap/sub-01_magnitude2.nii",
#         "/sub-01/fmap/sub-01_fieldmap.nii.gz",
#         "/sub-01/fmap/sub-01_fieldmap.json",
#         "/sub-01/fmap/sub-01_fieldmap.nii",
#         "/sub-01/fmap/sub-01_run-01_phasediff.nii.gz",
#         "/sub-01/fmap/sub-01_run-01_phasediff.json",
#         "/sub-01/fmap/sub-01_run-01_phasediff.nii",
#         "/sub-01/fmap/sub-01_run-01_phase1.nii.gz",
#         "/sub-01/fmap/sub-01_run-01_phase1.json",
#         "/sub-01/fmap/sub-01_run-01_phase1.nii",
#         "/sub-01/fmap/sub-01_run-01_phase2.nii.gz",
#         "/sub-01/fmap/sub-01_run-01_phase2.json",
#         "/sub-01/fmap/sub-01_run-01_phase2.nii",
#         "/sub-01/fmap/sub-01_run-01_magnitude.nii.gz",
#         "/sub-01/fmap/sub-01_run-01_magnitude.json",
#         "/sub-01/fmap/sub-01_run-01_magnitude.nii",
#         "/sub-01/fmap/sub-01_run-01_magnitude1.nii.gz",
#         "/sub-01/fmap/sub-01_run-01_magnitude1.json",
#         "/sub-01/fmap/sub-01_run-01_magnitude1.nii",
#         "/sub-01/fmap/sub-01_run-01_magnitude2.nii.gz",
#         "/sub-01/fmap/sub-01_run-01_magnitude2.json",
#         "/sub-01/fmap/sub-01_run-01_magnitude2.nii",
#         "/sub-01/fmap/sub-01_run-01_fieldmap.nii.gz",
#         "/sub-01/fmap/sub-01_run-01_fieldmap.json",
#         "/sub-01/fmap/sub-01_run-01_fieldmap.nii",
#         "/sub-01/fmap/sub-01_dir-dirlabel_epi.nii.gz",
#         "/sub-01/fmap/sub-01_dir-dirlabel_epi.json",
#         "/sub-01/fmap/sub-01_dir-dirlabel_epi.nii",
#         "/sub-01/fmap/sub-01_dir-dirlabel_run-01_epi.nii.gz",
#         "/sub-01/fmap/sub-01_dir-dirlabel_run-01_epi.json",
#         "/sub-01/fmap/sub-01_dir-dirlabel_run-01_epi.nii",
#         "/sub-01/fmap/sub-01_acq-singleband_phasediff.nii.gz",
#         "/sub-01/fmap/sub-01_acq-singleband_phasediff.json",
#         "/sub-01/fmap/sub-01_acq-singleband_phasediff.nii",
#         "/sub-01/fmap/sub-01_acq-singleband_phase1.nii.gz",
#         "/sub-01/fmap/sub-01_acq-singleband_phase1.json",
#         "/sub-01/fmap/sub-01_acq-singleband_phase1.nii",
#         "/sub-01/fmap/sub-01_acq-singleband_phase2.nii.gz",
#         "/sub-01/fmap/sub-01_acq-singleband_phase2.json",
#         "/sub-01/fmap/sub-01_acq-singleband_phase2.nii",
#         "/sub-01/fmap/sub-01_acq-singleband_magnitude.nii.gz",
#         "/sub-01/fmap/sub-01_acq-singleband_magnitude.json",
#         "/sub-01/fmap/sub-01_acq-singleband_magnitude.nii",
#         "/sub-01/fmap/sub-01_acq-singleband_magnitude1.nii.gz",
#         "/sub-01/fmap/sub-01_acq-singleband_magnitude1.json",
#         "/sub-01/fmap/sub-01_acq-singleband_magnitude1.nii",
#         "/sub-01/fmap/sub-01_acq-singleband_magnitude2.nii.gz",
#         "/sub-01/fmap/sub-01_acq-singleband_magnitude2.json",
#         "/sub-01/fmap/sub-01_acq-singleband_magnitude2.nii",
#         "/sub-01/fmap/sub-01_acq-singleband_fieldmap.nii.gz",
#         "/sub-01/fmap/sub-01_acq-singleband_fieldmap.json",
#         "/sub-01/fmap/sub-01_acq-singleband_fieldmap.nii",
#         "/sub-01/fmap/sub-01_acq-singleband_run-01_phasediff.nii.gz",
#         "/sub-01/fmap/sub-01_acq-singleband_run-01_phasediff.json",
#         "/sub-01/fmap/sub-01_acq-singleband_run-01_phasediff.nii",
#         "/sub-01/fmap/sub-01_acq-singleband_run-01_phase1.nii.gz",
#         "/sub-01/fmap/sub-01_acq-singleband_run-01_phase1.json",
#         "/sub-01/fmap/sub-01_acq-singleband_run-01_phase1.nii",
#         "/sub-01/fmap/sub-01_acq-singleband_run-01_phase2.nii.gz",
#         "/sub-01/fmap/sub-01_acq-singleband_run-01_phase2.json",
#         "/sub-01/fmap/sub-01_acq-singleband_run-01_phase2.nii",
#         "/sub-01/fmap/sub-01_acq-singleband_run-01_magnitude.nii.gz",
#         "/sub-01/fmap/sub-01_acq-singleband_run-01_magnitude.json",
#         "/sub-01/fmap/sub-01_acq-singleband_run-01_magnitude.nii",
#         "/sub-01/fmap/sub-01_acq-singleband_run-01_magnitude1.nii.gz",
#         "/sub-01/fmap/sub-01_acq-singleband_run-01_magnitude1.json",
#         "/sub-01/fmap/sub-01_acq-singleband_run-01_magnitude1.nii",
#         "/sub-01/fmap/sub-01_acq-singleband_run-01_magnitude2.nii.gz",
#         "/sub-01/fmap/sub-01_acq-singleband_run-01_magnitude2.json",
#         "/sub-01/fmap/sub-01_acq-singleband_run-01_magnitude2.nii",
#         "/sub-01/fmap/sub-01_acq-singleband_run-01_fieldmap.nii.gz",
#         "/sub-01/fmap/sub-01_acq-singleband_run-01_fieldmap.json",
#         "/sub-01/fmap/sub-01_acq-singleband_run-01_fieldmap.nii",
#         "/sub-01/fmap/sub-01_acq-singleband_dir-dirlabel_epi.nii.gz",
#         "/sub-01/fmap/sub-01_acq-singleband_dir-dirlabel_epi.json",
#         "/sub-01/fmap/sub-01_acq-singleband_dir-dirlabel_epi.nii",
#         "/sub-01/fmap/sub-01_acq-singleband_dir-dirlabel_run-01_epi.nii.gz",
#         "/sub-01/fmap/sub-01_acq-singleband_dir-dirlabel_run-01_epi.json",
#         "/sub-01/fmap/sub-01_acq-singleband_dir-dirlabel_run-01_epi.nii",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_phasediff.nii.gz",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_phasediff.json",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_phasediff.nii",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_phase1.nii.gz",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_phase1.json",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_phase1.nii",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_phase2.nii.gz",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_phase2.json",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_phase2.nii",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_magnitude.nii.gz",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_magnitude.json",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_magnitude.nii",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_magnitude1.nii.gz",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_magnitude1.json",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_magnitude1.nii",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_magnitude2.nii.gz",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_magnitude2.json",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_magnitude2.nii",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_fieldmap.nii.gz",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_fieldmap.json",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_fieldmap.nii",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_run-01_phasediff.nii.gz",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_run-01_phasediff.json",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_run-01_phasediff.nii",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_run-01_phase1.nii.gz",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_run-01_phase1.json",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_run-01_phase1.nii",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_run-01_phase2.nii.gz",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_run-01_phase2.json",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_run-01_phase2.nii",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_run-01_magnitude.nii.gz",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_run-01_magnitude.json",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_run-01_magnitude.nii",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_run-01_magnitude1.nii.gz",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_run-01_magnitude1.json",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_run-01_magnitude1.nii",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_run-01_magnitude2.nii.gz",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_run-01_magnitude2.json",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_run-01_magnitude2.nii",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_run-01_fieldmap.nii.gz",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_run-01_fieldmap.json",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_run-01_fieldmap.nii",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_dir-dirlabel_epi.nii.gz",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_dir-dirlabel_epi.json",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_dir-dirlabel_epi.nii",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_dir-dirlabel_run-01_epi.nii.gz",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_dir-dirlabel_run-01_epi.json",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_dir-dirlabel_run-01_epi.nii",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq-singleband_phasediff.nii.gz",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq-singleband_phasediff.json",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq-singleband_phasediff.nii",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq-singleband_phase1.nii.gz",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq-singleband_phase1.json",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq-singleband_phase1.nii",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq-singleband_phase2.nii.gz",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq-singleband_phase2.json",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq-singleband_phase2.nii",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq-singleband_magnitude.nii.gz",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq-singleband_magnitude.json",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq-singleband_magnitude.nii",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq-singleband_magnitude1.nii.gz",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq-singleband_magnitude1.json",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq-singleband_magnitude1.nii",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq-singleband_magnitude2.nii.gz",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq-singleband_magnitude2.json",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq-singleband_magnitude2.nii",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq-singleband_fieldmap.nii.gz",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq-singleband_fieldmap.json",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq-singleband_fieldmap.nii",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq-singleband_run-01_phasediff.nii.gz",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq-singleband_run-01_phasediff.json",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq-singleband_run-01_phasediff.nii",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq-singleband_run-01_phase1.nii.gz",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq-singleband_run-01_phase1.json",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq-singleband_run-01_phase1.nii",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq-singleband_run-01_phase2.nii.gz",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq-singleband_run-01_phase2.json",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq-singleband_run-01_phase2.nii",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq-singleband_run-01_magnitude.nii.gz",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq-singleband_run-01_magnitude.json",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq-singleband_run-01_magnitude.nii",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq-singleband_run-01_magnitude1.nii.gz",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq-singleband_run-01_magnitude1.json",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq-singleband_run-01_magnitude1.nii",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq-singleband_run-01_magnitude2.nii.gz",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq-singleband_run-01_magnitude2.json",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq-singleband_run-01_magnitude2.nii",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq-singleband_run-01_fieldmap.nii.gz",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq-singleband_run-01_fieldmap.json",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq-singleband_run-01_fieldmap.nii",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq-singleband_dir-dirlabel_epi.nii.gz",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq-singleband_dir-dirlabel_epi.json",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq-singleband_dir-dirlabel_epi.nii",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq-singleband_dir-dirlabel_run-01_epi.nii.gz",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq-singleband_dir-dirlabel_run-01_epi.json",
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq-singleband_dir-dirlabel_run-01_epi.nii"]

#     for item in target_list:
#         result = testvalidator.is_field_map(item)
#         assert result


# # checks is_field_map() function false cases
# def test_is_field_map_false(testvalidator):
#     target_list = [
#         "/sub-01/ses-test/fmap/sub--01_ses-test_acq-singleband_run-01_magnitude.json",  # wrong --
#         "/sub-01/ses-test/fmap/sub-01_ses-test__acq-singleband_run-01_magnitude.nii",  # wrong __
#         "/sub-01/ses-test/fmap/sub-01-ses-test_acq-singleband_run-01_magnitude1.nii.gz",  # wrong 01-ses
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq_singleband_run-01_magnitude1.json", # wrong acq_singleband
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq-singleband_run-01_magnitude1.ni", # wrong extension
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq-singleband_run-01_magnitude3.nii.gz", # wrong modality
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq-singeband_ran-01_magnitude2.json", # wrong ssuffix ran
#         "/sub-01/ses-test/fmap/sub-01_acq-singleband_run-01_magnitude2.nii", # missed session id in the filename
#         "/sub-01/ses-test/fmap/ses-test_acq-singleband_run-01_fieldmap.nii.gz", # missed subject id in the filename
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq-singleband_run-01.json",  # missed modality
#         "/sub-01/ses-test/fmap/sub-01_ses-test_acq-singleband_run-01_fieldmap",  # wrong extension
#         "/sub-01/fmap/ses-test/sub-01_ses-test_acq-singleband_dir-dirlabel_epi.nii.gz", # wrong dirs order
#         "/ses-test/fmap/sub-01/sub-01_ses-test_acq-singleband_dir-dirlabel_epi.json", # wrong dirs order
#         "/ses-test/sub-01/fmap/sub-01_ses-test_acq-singleband_dir-dirlabel_epi.nii", # wrong dirs order
#         "/sub-01/ses-test/sub-01_ses-test_acq-singleband_dir-dirlabel_run-01_epi.nii.gz", # missed data type dir
#         "/sub-01/fmap/sub-01_ses-test_acq-singleband_dir-dirlabel_run-01_epi.json", # missed session dir
#         "/ses-test/fmap/sub-01_ses-test_acq-singleband_dir-dirlabel_run-01_epi.nii" # missed subject dir
#     ]

#     for item in target_list:
#         result = testvalidator.is_field_map(item)
#         assert not result


# checks is_phenotypic() function true cases
def test_is_phenotypic_true(testvalidator):
    target_list = [
        "/phenotype/measurement_tool_name.tsv",
        "/phenotype/measurement_tool_name.json"
    ]

    for item in target_list:
        result = testvalidator.is_phenotypic(item)
        assert result


# checks is_phenotypic() function true cases
def test_is_phenotypic_false(testvalidator):
    target_list = [
        "/measurement_tool_name.tsv",  # missed phenotype dir
        "/phentype/measurement_tool_name.josn"  # wrong phenotype dir
        "/phenotype/measurement_tool_name.jsn"  # wrong extension
    ]

    for item in target_list:
        result = testvalidator.is_phenotypic(item)
        assert not result


def test_index_associated_false(testvalidator):
    testvalidator = BIDSValidator(index_associated=False)
    target_list = [
        "/code/",
        "/derivatives/",
        "/sourcedata/",
        "/stimuli/",
        "/.git/"
    ]

    for item in target_list:
        result = testvalidator.is_associated_data(item)
        assert not result

def test_layout_with_validation():
    data_dir = join(get_test_data_path(), '7t_trt')
    layout1 = BIDSLayout(data_dir, validate=True)
    layout2 = BIDSLayout(data_dir, validate=False)
    assert len(layout1.files) < len(layout2.files)
    # Not a valid BIDS file
    badfile = join(data_dir, 'test.bval')
    assert(badfile not in layout1.files)
    assert(badfile in layout2.files)
