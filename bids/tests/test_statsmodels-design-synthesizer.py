#!/usr/bin/env python

"""Tests for `bids_statsmodels_design_synthesizer` package."""

import pytest
import subprocess as sp
from pathlib import Path

SYNTHESIZER = "statsmodels-design-synthesizer"
from bids.statsmodels_design_synthesizer import statsmodels_design_synthesizer as synth_mod

# from bids_statsmodels_design_synthesizer import Path(SYNTHESIZER).stem as synth_mod
EXAMPLE_USER_ARGS = {
        "OUTPUT_TSV": "aggregated_design.tsv",
        "TRANSFORMS": "data/ds005/models/ds-005_type-mfx_model.json",
        "EVENTS_TSV": "data/ds005/sub-01/func/sub-01_task-mixedgamblestask_run-01_events.tsv",
        "TR": 2,
        "TA": 2,
        "NVOLS": 160,
    }


def test_cli_help():
    with pytest.raises(sp.CalledProcessError):
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
    bids_dir = Path(__file__).parent / "data/ds000003"
    model = "model-001_smdl.json"
    arg_list = " " .join([f"""--{k.lower().replace("_","-")}={v}""" for k,v in EXAMPLE_USER_ARGS.items()])
    cmd = f"{SYNTHESIZER} {arg_list}"
    output = sp.check_output(cmd.split())

