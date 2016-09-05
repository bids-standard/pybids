from bids.grabbids import BIDSLayout
from os.path import join, dirname

""" Tests of BIDS-specific functionality. Generic tests of core grabbit
functionality should go in the grabbit package. """


def test_layout_init():
    data_dir = join(dirname(__file__), 'data', '7t_trt')
    layout = BIDSLayout(data_dir)
    assert isinstance(layout.files, dict)


def test_get_metadata():
    data_dir = join(dirname(__file__), 'data', '7t_trt')
    layout = BIDSLayout(data_dir)
    target = 'sub-03/ses-2/func/sub-03_ses-2_task-' \
             'rest_acq-fullbrain_run-2_bold.nii.gz'
    result = layout.get_metadata(join(data_dir, target))
    assert result['RepetitionTime'] == 3.0


def test_get_metadata2():
    data_dir = join(dirname(__file__), 'data', '7t_trt')
    layout = BIDSLayout(data_dir)
    target = 'sub-03/ses-2/fmap/sub-03_ses-1_run-1_phasediff.nii.gz'
    result = layout.get_metadata(join(data_dir, target))
    assert result['EchoTime1'] == 0.006


def test_get_subjects():
    data_dir = join(dirname(__file__), 'data', '7t_trt')
    layout = BIDSLayout(data_dir)
    result = layout.get_subjects()
    predicted = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
    assert predicted == result


def test_get_fieldmap():
    data_dir = join(dirname(__file__), 'data', '7t_trt')
    layout = BIDSLayout(data_dir)
    target = 'sub-03/ses-2/func/sub-03_ses-2_task-' \
             'rest_acq-fullbrain_run-1_bold.nii.gz'
    result = layout.get_fieldmap(join(data_dir, target))
    assert result["type"] == "phasediff"
    assert result["phasediff"].endswith('sub-03_ses-2_run-1_phasediff.nii.gz')
