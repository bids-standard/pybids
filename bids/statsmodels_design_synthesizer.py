#! /usr/bin/env python
import argparse
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
from collections import namedtuple
from bids.modeling import transformations
from bids.utils import convert_JSON
from bids.variables import BIDSRunVariableCollection, SparseRunVariable
from bids.layout.utils import parse_file_entities
from bids.variables.io import get_events_collection
from bids.variables.entities import RunNode


def statsmodels_design_synthesizer(params):
    """Console script for bids statsmodels_design_synthesizer."""

    # Output:
    if not params.get("output_dir"):
        output_tsv = params.get("output_tsv", "aggregated_statsmodels_design.tsv")

    # Sampling rate of output
    sampling_rate_out = params.get("output_sampling_rate")

    # Process transformations file
    # TODO: add transforms functionality, for now only model.json is handled
    # TODO: some basic error checking to confirm the correct level of
    # transformations has been obtained. This will most likely be the case since
    # transformations at higher levels will no longer be required when the new
    # "flow" approach is used.
    transforms_file = Path(params["transforms"])
    if not transforms_file.exists():
        raise ValueError(f"Cannot find {transforms_file}")
    model = convert_JSON(json.loads(transforms_file.read_text()))

    if "nodes" in model:
        nodes_key = "nodes"
    elif "steps" in model:
        nodes_key = "steps"
    else:
        raise ValueError("Cannot find a key for nodes in the model file")
    model_transforms = model[nodes_key][0]["transformations"]

    duration = params["nvol"] * params["tr"]

    # Get relevant collection
    coll_df = pd.read_csv(params["events_tsv"], delimiter="\t")
    RunInfo = namedtuple("RunInfo", ["entities", "duration"])

    #run_info = RunInfo(parse_file_entities(params["events_tsv"]), duration)
    run = RunNode(parse_file_entities(params["events_tsv"]), None, duration, params["tr"], params["nvol"])
    coll = get_events_collection(coll_df, run, output='collection')

    # perform transformations
    colls = transformations.TransformerManager().transform(coll, model_transforms)

    # Save colls
    df_out = colls.to_df(sampling_rate=sampling_rate_out)
    df_out.to_csv(output_tsv, index=None, sep="\t", na_rep="n/a")

def create_parser():
    """Returns argument parser"""
    p = argparse.ArgumentParser()
    p.add_argument("--events-tsv", required=True, help="Path to events TSV")
    p.add_argument(
        "--transforms", required=True, help="Path to transform or model json"
    )
    p.add_argument(
        "--output-sampling-rate",
        required=False,
        type=float,
        help="Output sampling rate in Hz when output is dense instead of sparse",
    )

    pout = p.add_mutually_exclusive_group()
    pout.add_argument(
        "--output-tsv",
        nargs="?",
        help="Path to TSV containing a fully constructed design matrix.",
    )
    pout.add_argument(
        "--output-dir",
        nargs="?",
        help="Path to directory to write processed event files.",
    )

    ptimes = p.add_argument_group(
        "Specify some essential details about the time series."
    )
    ptimes.add_argument(
        "--nvol", required=True, type=int, help="Number of volumes in func time-series"
    )
    ptimes.add_argument(
        "--tr", required=True, type=float, help="TR for func time series"
    )
    ptimes.add_argument("--ta", required=True, type=float, help="TA for events")

    return p


def main(user_args=None):
    parser = create_parser()
    if user_args is None:
        namespace = parser.parse_args(sys.argv[1:])
        params = vars(namespace)
    else:
        params = user_args

    statsmodels_design_synthesizer(params)


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover""Main module."""
