''' Test-related utilities '''

from pathlib import Path


def get_test_data_path():
    return str(Path(__file__).parent.parent.parent.parent / 'tests' / 'data')
