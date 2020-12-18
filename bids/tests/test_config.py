import bids
import tempfile
import os
import json
import pytest
from bids.config import reset_options
from bids.tests import get_test_data_path


def test_load_from_standard_paths():
    # Verify defaults
    reset_options(False)
    assert bids.config._settings == bids.config._default_settings

    env_config = {"loop_preproc": True}
    cwd_config = {"loop_preproc": True}

    handle, f = tempfile.mkstemp(suffix='.json')
    json.dump(env_config, open(f, 'w'))
    os.environ['PYBIDS_CONFIG'] = f
    target = 'pybids_config.json'
    if os.path.exists(target):
        pytest.skip("Cannot test bids config because the default config file"
                    " (pybids_config.json) already exists in the current "
                    "working directory. Skipping test to avoid overwriting.")
    json.dump(cwd_config, open(target, 'w'))
    reset_options(True)
    os.unlink(target)
    opts = bids.config._settings

    # assert opts['loop_preproc']

    reset_options(False)


def test_set_option():

    reset_options(False)
    opts = bids.config._settings

    # bids.config.set_options(loop_preproc=False)
    # assert opts['loop_preproc'] == False

    with pytest.raises(ValueError):
        bids.config.set_option('bad_key', False)

    reset_options(False)


# def test_get_option():
#     reset_options(False)
#     assert not bids.config.get_option('loop_preproc')


def test_extension_initial_dot(mock_config):
    ds117 = os.path.join(get_test_data_path(), 'ds000117')

    # Warn if creating a layout without declaring a mode
    bids.config.set_option('extension_initial_dot', None)
    with pytest.warns(FutureWarning, match='To suppress this warning'):
        layout = bids.BIDSLayout(ds117)
    assert layout.get(extension='nii.gz')[0].entities['extension'] == 'nii.gz'

    # Warn if setting the mode to False... this isn't sticking around
    with pytest.warns(FutureWarning, match='will be disabled'):
        bids.config.set_option('extension_initial_dot', False)
    with pytest.warns(None) as record:
        layout = bids.BIDSLayout(ds117)
    assert len(record) == 0
    assert layout.get(extension='nii.gz')[0].entities['extension'] == 'nii.gz'

    # No warnings to move to dot mode
    with pytest.warns(None) as record:
        bids.config.set_option('extension_initial_dot', True)
        layout = bids.BIDSLayout(ds117)
    assert len(record) == 0
    assert layout.get(extension='nii.gz')[0].entities['extension'] == '.nii.gz'
