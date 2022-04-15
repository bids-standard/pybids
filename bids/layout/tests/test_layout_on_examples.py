""" Tests runs layout on bids examples and make sure all files are caught"""

from os.path import join

import pytest

from bids.layout import BIDSLayout
from bids.tests import get_test_data_path

# Values for the number of file got from a:
#
#   find ds_name -type f | wc -l
#
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
        ("qmri_qsm", 8),
        ("qmri_vfa", 17),
    ],
)
def test_index_metadata(dataset, nb_files):
    ds = join(get_test_data_path(), "bids-examples", dataset)
    layout = BIDSLayout(ds, derivatives=True)
    files = layout.get()
    assert len(files) == nb_files
