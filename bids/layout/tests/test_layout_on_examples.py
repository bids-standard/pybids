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
        ("qmri_irt1", 15),
        ("qmri_mese", 73),
        ("qmri_mp2rage", 14),
        ("qmri_mp2rageme", 28),
        ("qmri_mtsat", 23),
        ("qmri_sa2rage", 9),
        ("qmri_vfa", 17),
        ("qmri_mpm", 125),
    ],
)
def test_layout_on_examples_with_derivatives(dataset, nb_files, bids_examples):
    layout = BIDSLayout(bids_examples / dataset, derivatives=True)
    files = layout.get()
    assert len(files) == nb_files


@pytest.mark.parametrize(
    "dataset, nb_files",
    [
        ("micr_SEM", 12),
        ("micr_SPIM", 22),
        ("asl001", 8),
        ("asl002", 10),
        ("asl003", 10),
        ("asl004", 12),
        ("asl005", 10),
        ("pet001", 12),
        ("pet002", 20),
        ("pet003", 9),
        ("pet004", 10),
        ("pet005", 14),
        ("qmri_megre", 18),
        ("qmri_tb1tfl", 6),
        ("qmri_qsm", 8),
    ],
)
def test_layout_on_examples_no_derivatives(dataset, nb_files, bids_examples):
    layout = BIDSLayout(bids_examples / dataset)
    files = layout.get()
    assert len(files) == nb_files