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


def statsmodels_design_synthesizer(params):
    """Console script for bids statsmodels_design_synthesizer."""

    # Output:
    if not params.get("OUTPUT_DIR"):
        output_tsv = params.get("OUTPUT_TSV", "aggregated_statsmodels_design.tsv")

    # Sampling rate of output
    sampling_rate_out = params.get("OUTPUT_SAMPLING_RATE")

    # Process transformations file
    # TODO: add transforms functionality, for now only model.json is handled
    # TODO: some basic error checking to confirm the correct level of
    # transformations has been obtained. This will most likely be the case since
    # transformations at higher levels will no longer be required when the new
    # "flow" approach is used.
    transforms_file = Path(params["TRANSFORMS"])
    if not transforms_file.exists():
        raise ValueError(f"Cannot find {transforms_file}")
    model = convert_JSON(json.loads(model_file.read_text()))
    model_transforms = model["steps"][0]["transformations"]

    # Get relevant collection
    coll_df = pd.read_csv(params["EVENTS_TSV"], delimiter="\t")
    RunInfo = namedtuple("RunInfo", ["entities", "duration"])
    run_info = RunInfo(parse_file_entities(params["EVENTS_TSV"]), params["DURATION"])
    coll = get_events_collection(coll_df, [run_info])

    # perform transformations
    colls = transformations.TransformerManager().transform(coll, model_transforms)

    # Save colls
    df_out = colls.to_df(sampling_rate=sampling_rate_out)
    df_out.to_csv(output_tsv, index=None, sep="\t", na_rep="n/a")


def get_events_collection(_data, run_info, drop_na=True):
    """ "
    This is an attempt to minimally implement:
    https://github.com/bids-standard/pybids/blob/statsmodels/bids/variables/io.py
    """
    colls_output = []
    if "amplitude" in _data.columns:
        if (
            _data["amplitude"].astype(int) == 1
        ).all() and "trial_type" in _data.columns:
            msg = (
                "Column 'amplitude' with constant value 1 "
                "is unnecessary in event files; ignoring it."
            )
            _data = _data.drop("amplitude", axis=1)
        else:
            msg = "Column name 'amplitude' is reserved; " "renaming it to 'amplitude_'."
            _data = _data.rename(columns={"amplitude": "amplitude_"})
            warnings.warn(msg)

    _data = _data.replace("n/a", np.nan)  # Replace BIDS' n/a
    _data = _data.apply(pd.to_numeric, errors="ignore")

    _cols = list(set(_data.columns.tolist()) - {"onset", "duration"})

    # Construct a DataFrame for each extra column
    for col in _cols:
        df = _data[["onset", "duration"]].copy()
        df["amplitude"] = _data[col].values

        # Add in all of the run's entities as new columns for
        # index
        #        for entity, value in entities.items():
        #            if entity in ALL_ENTITIES:
        #                df[entity] = value
        #
        if drop_na:
            df = df.dropna(subset=["amplitude"])

        if df.empty:
            continue
        var = SparseRunVariable(name=col, data=df, run_info=run_info, source="events")
        colls_output.append(var)

    output = BIDSRunVariableCollection(colls_output)
    return output


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
        "--nvol", required=True, help="Number of volumes in func time-series"
    )
    ptimes.add_argument("--tr", required=True, help="TR for func time series")
    ptimes.add_argument("--ta", required=True, help="TA for events")

    return p


def main(user_args=None):
    parser = create_parser()
    if user_args is None:
        namespace = parser.parse_args(sys.argv[1:])
    else:
        namespace = parser.parse_args(user_args)
    params = vars(namespace)

    statsmodels_design_synthesizer(params)


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover""Main module."""
