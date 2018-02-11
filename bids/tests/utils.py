''' Test-related utilities '''

from os.path import join, dirname


def get_test_data_path():
    return join(dirname(__file__), 'data')
