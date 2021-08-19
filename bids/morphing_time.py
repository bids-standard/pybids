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
from bids.variables.io import get_events_collection, parse_transforms
from bids.variables.entities import RunNode


def  morphing_time(
    *,
    events_tsv,
    transforms,
    nvol,
    tr,
    ta=None,
    output_sampling_rate=None,
    output_dir=None,
 ):

    output_dir = Path(output_dir  or "design_synthesizer")
    output_dir.mkdir(exist_ok=True) 
    model_transforms = parse_transforms(transforms)
    duration = nvol * tr
    ta = ta or tr

    # Get relevant collection
    coll_df = pd.read_csv(events_tsv, delimiter="\t")
    RunInfo = namedtuple('RunInfo', ['entities', 'duration', 'tr', 'image', 'n_vols'])

    #run_info = RunInfo(parse_file_entities(events_tsv), duration)
    # TODO: this will need to be implemented without RunNode to break cyclic
    # dependencies if transformations is to be extracted
    run_info = RunInfo(parse_file_entities(events_tsv), duration, tr, None, nvol)
    coll = BIDSRunVariableCollection(get_events_collection(coll_df, run_info))

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

