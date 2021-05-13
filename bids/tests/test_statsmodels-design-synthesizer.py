#!/usr/bin/env python

"""Tests for `bids_statsmodels_design_synthesizer` package."""

import pytest
import subprocess as sp
from pathlib import Path
import tempfile

SYNTHESIZER = "statsmodels-design-synthesizer"
from bids import statsmodels_design_synthesizer as synth_mod

# from bids_statsmodels_design_synthesizer import Path(SYNTHESIZER).stem as synth_mod
DATA_DIR = (Path(__file__).parent / "data/ds005").absolute()

# Define some example user arg combinations (without output_dir which is better
# to define in the scope of the test)
EXAMPLE_USER_ARGS = {
        "transforms": f"{DATA_DIR}/models/ds-005_type-mfx_model.json",
        "events_tsv": f"{DATA_DIR}/sub-01/func/sub-01_task-mixedgamblestask_run-01_events.tsv",
        "tr": 2,
        "ta": 2,
        "nvol": 160,
    }
EXAMPLE_USER_ARGS_2 = {
        "transforms": f"{DATA_DIR}/models/ds-005_type-test_model.json",
        "events_tsv": f"{DATA_DIR}/sub-01/func/sub-01_task-mixedgamblestask_run-01_events.tsv",
        "tr": 2,
        "ta": 2,
        "nvol": 160,
        "output_sampling_rate":10,
    }
EXAMPLE_USER_ARGS_3 = EXAMPLE_USER_ARGS_2.copy()
EXAMPLE_USER_ARGS_3["transforms"] = f"{DATA_DIR}/models/ds-005_type-convolution_model.json"


def test_cli_help():
    output = sp.check_output([SYNTHESIZER, "-h"])
    with pytest.raises(sp.CalledProcessError):
        output = sp.check_output([SYNTHESIZER, "--non-existent"])


@pytest.mark.parametrize(
    "test_case,user_args",
    [
        ("Model type test", EXAMPLE_USER_ARGS),
        ("Model type mfx", EXAMPLE_USER_ARGS_2),
        ("Model type convolution", EXAMPLE_USER_ARGS_3),
    ]
)
def test_design_aggregation_function(tmp_path,test_case,user_args):
    user_args['output_dir'] = str(tmp_path)
    synth_mod.main(user_args)

@pytest.mark.parametrize(
    "test_case,user_args",
    [
        ("Model type test", EXAMPLE_USER_ARGS),
        ("Model type mfx", EXAMPLE_USER_ARGS_2),
    ]
)
def test_minimal_cli_functionality(tmp_path,test_case,user_args):
    """
    We roughly want to implement the equivalent of the following:
    from bids.analysis import Analysis
    from bids.layout import BIDSLayout

    layout = BIDSLayout("data/ds000003")
    analysis = Analysis(model="data/ds000003/models/model-001_smdl.json",layout=layout)
    analysis.setup()

    more specifically we want to reimplement this line
    https://github.com/bids-standard/pybids/blob/b6cd0f6787230ce976a374fbd5fce650865752a3/bids/analysis/analysis.py#L282
    """
    user_args['output_dir'] = str(tmp_path)
    arg_list = " " .join([f"""--{k.lower().replace("_","-")}={v}""" for k,v in user_args.items()])
    cmd = f"{SYNTHESIZER} {arg_list}"
    output = sp.check_output(cmd.split())


