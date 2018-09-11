import bids
import tempfile
import os
import json
import pytest
from bids.config import reset_options


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
