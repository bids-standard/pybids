from bids.analysis.variables import BIDSVariableManager
import pytest
from os.path import join, dirname, abspath
from bids import grabbids


@pytest.fixture
def manager():
    mod_file = abspath(grabbids.__file__)
    path = join(dirname(mod_file), 'tests', 'data', 'ds005')
    return BIDSVariableManager(path)


def test_get_design_matrix(manager):
    manager.load()
    manager.set_analysis_level('run')
    subs = [str(s).zfill(2) for s in [1, 2, 3, 4, 5, 6]]
    dm = manager.get_design_matrix(columns=['RT', 'parametric gain'],
                                   subject=subs)
    assert dm.shape == (4308, 6)


def test_analysis_smoke_test(manager):
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
