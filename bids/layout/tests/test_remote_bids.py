""" Tests runs layout on bids examples and make sure all files are caught"""

""" TODO
- add more 'vanilla' datasets
- missing files in micr?
"""

import pytest

from bids.layout import BIDSLayout

# Values for the number of files by downloading dataset first

@pytest.mark.parametrize(
    "dataset, nb_files",
    [
        ("s3://openneuro.org/ds000102", 136),
    ],
)
def test_layout_on_s3_datasets_no_derivatives(dataset, nb_files):
    layout = BIDSLayout(dataset)
    files = layout.get()
    assert len(files) == nb_files



