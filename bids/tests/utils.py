''' Test-related utilities '''

from os.path import join, dirname, abspath
from pathlib import Path


def get_test_data_path():
    return join(dirname(abspath(__file__)), 'data')
