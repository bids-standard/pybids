from bids.variables import load_variables
from collections import OrderedDict
import numpy as np


def _make_passthrough_contrast(level, model_type='glm', test="t"):
    gb_dict = dict(Subject=['subject', 'contrast'],
                   Session=['session', 'contrast'],
                   Dataset=['contrast'])
    block = dict(
        Level=level,
        Name=level,
        GroupBy=gb_dict[level],
        Model={'Type': model_type, 'X': [1]},
        DummyContrasts={"Test": test}
    )
    return block


def auto_model(layout, scan_length=None, one_vs_rest=False):
    """Create a simple default model for each of the tasks in a BIDSLayout.
    Contrasts each trial type against all other trial types and trial types
    at the run level and then uses dummy contrasts at each other level
    present to aggregate these results up.

    Parameters
    ----------
    layout : :obj:`bids.layout.BIDSLayout`
        A BIDSLayout instance
    scan_length : int
        Scan length for loading event variables in cases
        where the scan length can not be read from the nifti.
        Primarily for testing.
    one_vs_rest : bool
        Set to True if you would like to autogenerate
        contrasts of each trial type against everyother trialtype.

    Returns
    -------
    list
        list of model dictionaries for each task
    """

    base_name = layout._root.name
    tasks = layout.entities['task'].unique()
    task_models = []

    for task_name in tasks:
        # Populate model meta-data
        model = OrderedDict()
        model["Name"] = "_".join([base_name, task_name])
        model["Description"] = ("Autogenerated model for the %s task from %s" %
                                (task_name, base_name))
        model["BIDSModelVersion"]= "1.0.0"
        model["Input"] = {"task": [task_name]}
        nodes = []

        # Make run level block
        transformations = dict(
            Transformer='pybids-transforms-v1',
            Instructions=[
                dict(
                    Name='Factor',
                    Input='trial_type'
                )
            ]
        )
        run = dict(Level='Run',
                   Name='Run',
                   GroupBy=['run', 'subject'],
                   Transformations=transformations)

        # Get trial types
        run_nodes = load_variables(layout, task=task_name, levels=['run'],
                                   scan_length=scan_length)

        evs = []
        for n in run_nodes.nodes:
            evs.extend(n.variables['trial_type'].values.values)
        trial_types = np.unique(evs)
        trial_type_factors = ["trial_type." + tt for tt in trial_types]

        run_model = dict(Type='glm', X=trial_type_factors)
        # Add HRF
        run_model['HRF'] = dict(
            Variables=trial_type_factors,
            Model="DoubleGamma",
            Parameters=dict(
                PeakDelay=3,
                PeakDispersion=6,
                UndershootDelay=10,
                UndershootDispersion=12,
                PeakUndershootRatio=0.2
            )
        )
        run["Model"] = run_model

        if one_vs_rest:
            # If there are multiple trial types, build contrasts
            contrasts = []
            for tt in trial_types:
                cdict = OrderedDict()
                if len(trial_types) > 1:
                    cdict["Name"] = "run_" + tt + "_vs_others"
                else:
                    cdict["Name"] = "run_" + tt
                cdict["ConditionList"] = trial_type_factors

                # Calculate weights for contrast
                weights = np.ones(len(trial_types))
                try:
                    weights[trial_types != tt] = -1.0 / (len(trial_types) - 1)
                except ZeroDivisionError:
                    pass
                cdict["Weights"] = list(weights)

                cdict["Test"] = "t"
                contrasts.append(cdict)

            run["Contrasts"] = contrasts
        nodes.append(run)

        if one_vs_rest:
            # if there are multiple sessions/subjects, t-test run level contrasts at
            # session/subject level
            if len(layout.get_sessions()) > 1:
                nodes.append(_make_passthrough_contrast("Session", "meta"))

            if len(layout.get_subjects()) > 1:
                nodes.append(_make_passthrough_contrast("Subject", "meta"))

            nodes.append(_make_passthrough_contrast("Dataset"))

        model["Nodes"] = nodes
        task_models.append(model)

    return task_models
