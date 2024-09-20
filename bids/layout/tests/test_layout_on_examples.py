""" Tests runs layout on bids examples and make sure all files are caught"""

""" TODO
- add more 'vanilla' datasets
- missing files in micr?
"""

import pytest

from bids.layout import BIDSLayout

# Values for the number of file got from a:
#
#   find ds_name -type f | wc -l
#

@pytest.mark.parametrize(
    "dataset, nb_files",
    [
        ("qmri_irt1", 17),
        ("qmri_mese", 74),
        ("qmri_mp2rage", 18),
        ("qmri_mp2rageme", 29),
        ("qmri_mtsat", 26),
        ("qmri_sa2rage", 10),
        ("qmri_vfa", 19),
        ("qmri_mpm", 126),
        ("qmri_qsm", 10),
    ],
)
def test_layout_on_examples_with_derivatives(dataset, nb_files, bids_examples):
    layout = BIDSLayout(bids_examples / dataset, derivatives=True)
    files = layout.get()
    assert len(files) == nb_files


@pytest.mark.parametrize(
    "dataset, nb_files",
    [
        ("micr_SEM", 16),
        ("micr_SPIM", 26),
        ("asl001", 8),
        ("asl002", 10),
        ("asl003", 10),
        ("asl004", 12),
        ("asl005", 10),
        ("ds001", 135), # with anat and func data
        ("eeg_cbm", 104),
        ("ds000246", 19), # with meg data (excludes all files in .ds folders)
        ("genetics_ukbb", 96), # with dwi data
        ("ieeg_visual_multimodal", 148), # also with fmap data; ignore stimmuli folder
        ("pet001", 12),
        ("pet002", 20),
        ("pet003", 9),
        ("pet004", 10),
        ("pet005", 14),
        ("qmri_megre", 19),
        ("qmri_tb1tfl", 6),
        ("motion_dualtask", 645),
        ("fnirs_tapping", 39),
        ("mrs_fmrs", 169),
        ("mrs_biggaba", 172),
        ("mrs_2dmrsi", 67),
    ],
)
def test_layout_on_examples_no_derivatives(dataset, nb_files, bids_examples):
    layout = BIDSLayout(bids_examples / dataset)
    files = layout.get()
    assert len(files) == nb_files