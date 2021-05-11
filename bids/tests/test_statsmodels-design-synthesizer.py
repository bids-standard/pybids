#!/usr/bin/env python

"""Tests for `bids_statsmodels_design_synthesizer` package."""

import pytest
import subprocess as sp
from pathlib import Path

SYNTHESIZER = "statsmodels-design-synthesizer"
from bids import statsmodels_design_synthesizer as synth_mod

# from bids_statsmodels_design_synthesizer import Path(SYNTHESIZER).stem as synth_mod
DATA_DIR = (Path(__file__).parent / "data/ds005").absolute()
EXAMPLE_USER_ARGS = {
        "output_tsv": "aggregated_design.tsv",
        "transforms": f"{DATA_DIR}/models/ds-005_type-mfx_model.json",
        "events_tsv": f"{DATA_DIR}/sub-01/func/sub-01_task-mixedgamblestask_run-01_events.tsv",
        "tr": 2,
        "ta": 2,
        "nvol": 160,
    }


def test_cli_help():
    output = sp.check_output([SYNTHESIZER, "-h"])
    with pytest.raises(sp.CalledProcessError):
        output = sp.check_output([SYNTHESIZER, "--non-existent"])


def test_design_aggregation_function():
    synth_mod.main(EXAMPLE_USER_ARGS)


def test_minimal_cli_functionality():
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
    arg_list = " " .join([f"""--{k.lower().replace("_","-")}={v}""" for k,v in EXAMPLE_USER_ARGS.items()])
    cmd = f"{SYNTHESIZER} {arg_list}"
    output = sp.check_output(cmd.split())

