"""Tests runs layout on bids examples and make sure all files are caught"""

# TODO
# - add more 'vanilla' datasets
# - missing files in micr?

import pytest
from botocore.exceptions import NoCredentialsError
from upath import UPath

from bids.layout import BIDSLayout

# Values for the number of files by downloading dataset first


@pytest.mark.parametrize(
    "dataset, nb_files",
    [
        (UPath("s3://openneuro.org/ds000102", anon=True), 136),
    ],
)
@pytest.mark.xfail(raises=NoCredentialsError)
def test_layout_on_s3_datasets_no_derivatives(dataset, nb_files):
    layout = BIDSLayout(dataset)
    files = layout.get()
    assert len(files) == nb_files
