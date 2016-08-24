from bids.grabbids import BIDSLayout
from os.path import join, dirname

""" Tests of BIDS-specific functionality. Generic tests of core grabbit
functionality should go in the grabbit package. """


def test_layout_init():
    data_dir = join(dirname(__file__), 'data', '7t_trt')
    layout = BIDSLayout(data_dir)
    assert isinstance(layout.files, dict)


def test_gwt_metadata():
    data_dir = join(dirname(__file__), 'data', '7t_trt')
    layout = BIDSLayout(data_dir)
    target = 'sub-03/ses-2/func/sub-03_ses-2_task-' \
             'rest_acq-fullbrain_run-2_bold.nii.gz'
    result = layout.get_metadata(join(data_dir, target))
    assert result['RepetitionTime'] == 3.0
