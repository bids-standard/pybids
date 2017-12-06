from os.path import join, dirname, abspath
from bids import grabbids


def test_analysis_smoke_test():
    from bids.analysis.base import Analysis
    mod_file = abspath(grabbids.__file__)
    layout_path = join(dirname(mod_file), 'tests', 'data', 'ds005')
    json_file = join(layout_path, 'models', 'ds-005_type-test_model.json')

    analysis = Analysis(json_file, layouts=layout_path)
    analysis.setup(apply_transformations=True)

    result = analysis['firstlevel'].get_Xy(subject=['01', '02'])
    assert len(result) == 6
    assert len(result[0]) == 3
    assert 'sub-01_task-mixedgamblestask_run-01_bold.nii.gz' in result[0][1]

    result = analysis['secondlevel'].get_Xy()
    assert len(result) == 16
    assert len(result[0]) == 3
    assert result[0].data.shape == (3, 8)
    assert result[0].image is None
    assert result[0].entities == {'subject': '01'}
