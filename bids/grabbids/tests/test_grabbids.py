from bids.grabbids import Layout
from os.path import join, dirname

""" Tests of BIDS-specific functionality. Generic tests of core grabbit
functionality should go in the grabbit package. """


def test_layout_init():
    data_dir = join(dirname(__file__), 'data', '7t_trt')
    layout = Layout(data_dir)
    assert isinstance(layout.files, dict)


def test_find_nearest():
    data_dir = join(dirname(__file__), 'data', '7t_trt')
    layout = Layout(data_dir)
    target = 'sub-03/ses-2/func/sub-03_ses-2_task-rest_acq-fullbrain_run-2_bold.nii.gz'
    result = layout.find_match('bval', join(data_dir, *target.split('/')))
    assert 'sub-03-test.bval' in result
