from os.path import join, dirname, abspath
from bids import grabbids
from bids.grabbids import BIDSLayout


def test_analysis_smoke_test():
    from bids.analysis.base import Analysis
    mod_file = abspath(grabbids.__file__)
    layout_path = join(dirname(mod_file), 'tests', 'data', 'ds005')
    json_file = join(layout_path, 'models', 'ds-005_type-test_model.json')

    analysis = Analysis(layout_path, json_file)
    analysis.setup(apply_transformations=True)

    result = analysis['firstlevel'].get_Xy(subject=['01', '02'])
    assert len(result) == 6
    assert len(result[0]) == 3
    assert 'sub-01_task-mixedgamblestask_run-01_bold.nii.gz' in result[0][1]

    result = analysis['secondlevel'].get_Xy()
    assert len(result) == 16
    assert len(result[0]) == 3
    assert result[0].data.shape == (24, 2)
    assert result[0].image is None
    assert result[0].entities == {'subject': 1}

    result = analysis['firstlevel'].get_Xy(drop_entities=False)
    assert len(result) == 48
    assert result[0][0].shape == (688, 7)
    assert set(result[0][0].columns) == {'amplitude', 'onset', 'duration',
                                         'condition', 'subject', 'run',
                                         'task'}
