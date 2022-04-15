""" Tests runs layout on bids examples and make sure all files are catched"""

import os
import re
from os.path import join, abspath, basename
from pathlib import Path
import shutil
import json

import numpy as np
import pytest

from bids.layout import BIDSLayout, Query
from bids.layout.models import Config
from bids.layout.index import BIDSLayoutIndexer
from bids.layout.utils import PaddedInt
from bids.tests import get_test_data_path
from bids.utils import natural_sort


def test_layout_init(layout_7t_trt):
    assert isinstance(layout_7t_trt.files, dict)


@pytest.mark.parametrize(
    'index_metadata,query,result',
    [
        (True, {}, 3.0),
        (False, {}, None),
        (True, {}, 3.0),
        (True, {'task': 'rest'}, 3.0),
        (True, {'task': 'rest', 'extension': ['.nii.gz']}, 3.0),
        (True, {'task': 'rest', 'extension': '.nii.gz'}, 3.0),
        (True, {'task': 'rest', 'extension': ['.nii.gz', '.json'], 'return_type': 'file'}, 3.0),
    ])
def test_index_metadata(index_metadata, query, result, mock_config):
    data_dir = join(get_test_data_path(), '7t_trt')
    layout = BIDSLayout(data_dir, index_metadata=index_metadata, **query)
    sample_file = layout.get(task='rest', extension='.nii.gz',
                             acquisition='fullbrain')[0]
    metadata = sample_file.get_metadata()
    assert metadata.get('RepetitionTime') == result

