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
from bids.variables import BIDSRunVariableCollection, SparseRunVariable, merge_collections
from bids.layout.utils import parse_file_entities
from bids.variables.io import get_events_collection
from bids.variables.entities import RunNode
import click

from . import __version__


@click.command()
@click.version_option(__version__, prog_name='statsmodels_design_sythesizer')
@click.option(
    "--events-tsv", required=True, help="Path to events TSV")
@click.option(
        "--transforms", required=True, help="Path to transform or model json"
    )
@click.option(
        "--nvol", required=True, type=int, help="Number of volumes in func time-series"
    )
@click.option(
        "--tr", required=True, type=float, help="TR for func time series"
    )
@click.option(
    "--ta", required=True, type=float, help="TA for events")
@click.option(
        "--output-sampling-rate",
        required=False,
        type=float,
        help="Output sampling rate in Hz when a full design matrix is desired.",
    )
@click.option(
        "--output-dir",
        required=False,
        help="Path to directory to write processed event files.",
    )
def main(**kwargs):
    statsmodels_design_synthesizer(**kwargs)

def  statsmodels_design_synthesizer(
    *,
    events_tsv,
    transforms,
    nvol,
    tr,
    ta,
    output_sampling_rate=None,
    output_dir=None,
 ):

    output_dir = Path(output_dir  or "design_synthesizer")
    output_dir.mkdir(exist_ok=True) 

    # Process transformations file
    # TODO: abstact transforms file reading into a function.
    # TODO: add transforms functionality, for now only model.json is handled
    # TODO: some basic error checking to confirm the correct level of
    # transformations has been obtained. This will most likely be the case since
    # transformations at higher levels will no longer be required when the new
    # "flow" approach is used.
    transforms_file = Path(transforms)
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

    duration = nvol * tr

    # Get relevant collection
    coll_df = pd.read_csv(events_tsv, delimiter="\t")
    RunInfo = namedtuple("RunInfo", ["entities", "duration"])

    #run_info = RunInfo(parse_file_entities(params["events_tsv"]), duration)
    # TODO: this will need to be implemented without RunNode to break cyclic
    # dependencies if transformations is to be extracted
    run = RunNode(parse_file_entities(events_tsv), None, duration, tr, nvol)
    coll = get_events_collection(coll_df, run, output='collection')

    # perform transformations, additionally save variables that were changed.
    # If a column is transformed but not densified it will not be in
    # colls_pre_densification.
    colls, colls_pre_densification = (
        transformations.TransformerManager(save_pre_dense=True)
        .transform(coll, model_transforms)
        )

    # Save sparse vars
    if colls_pre_densification is not None:
        final_sparse_colls = BIDSRunVariableCollection(colls.get_sparse_variables())
        final_sparse_names = set([vv for vv in final_sparse_colls.variables])
        pre_dense_names = set([vv for vv in colls_pre_densification.variables])
        shared_names = final_sparse_names.intersection(pre_dense_names)

        if len(shared_names) > 0:
            raise ValueError(
        f"""Somehow you've ended up with a copy of {shared_names} in both the final
        transformed variables and in the pre-densification variables. Did you delete a
        variable and recreate one with same name?"""
        )
        output = merge_collections(
            [colls_pre_densification, final_sparse_colls]
        )
        assert output.all_sparse()

        df_sparse = output.to_df()
    else:
        df_sparse = colls.to_df(include_dense=False)

    df_sparse.to_csv(output_dir / "transformed_events.tsv", index=None, sep="\t", na_rep="n/a")
    # Save dense vars
    try:
        df_dense = colls.to_df(include_sparse=False)
        df_dense.to_csv(output_dir / "transformed_time_series.tsv", index=None, sep="\t", na_rep="n/a")
    except ValueError:
        pass

    # Save full design_matrix
    if output_sampling_rate:
        df_full = colls.to_df(sampling_rate=output_sampling_rate)
        df_full.to_csv(output_dir / "aggregated_design.tsv", index=None, sep="\t", na_rep="n/a")



if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover""Main module."""
