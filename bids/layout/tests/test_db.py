"""Test functionality in the db module--mostly related to connection
management."""

import re
from pathlib import Path

from bids.layout.db import (ConnectionManager, get_database_file,
                            get_database_sidecar, _sanitize_init_args)


def test_sanitize_init_args():
    patt = [re.compile('f1ct.*n'), 'code']
    root = Path('.') / 'fictional_path'
    result = _sanitize_init_args(ignore=patt, root=root, absolute_paths=False)
    assert result['absolute_paths'] == False
    assert isinstance(result['ignore'], list)
    assert all([isinstance(el, str) for el in result['ignore']])
    assert isinstance(result['root'], str)
