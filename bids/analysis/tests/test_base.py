from bids.analysis.variables import BIDSVariableManager
import pytest
from os.path import join, dirname, abspath
from bids import grabbids


@pytest.fixture
def manager():
    mod_file = abspath(grabbids.__file__)
    path = join(dirname(mod_file), 'tests', 'data', 'ds005')
    return BIDSVariableManager(path)


def test_analysis_smoke_test():
    from bids.analysis.base import Analysis
    mod_file = abspath(grabbids.__file__)
    layout_path = join(dirname(mod_file), 'tests', 'data', 'ds005')
    json_file = join(layout_path, 'models', 'ds-005_type-test_model.json')

    analysis = Analysis(layout_path, json_file)
    analysis.setup()

    result = analysis['secondlevel'].get_Xy()
    assert len(result) == 16
    assert len(result[0]) == 3
    assert result[0][0].shape == (3, 8)
