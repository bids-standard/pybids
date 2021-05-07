from os.path import join
from itertools import chain

import numpy as np
import pytest

from bids.modeling import BIDSStatsModelsGraph
from bids.modeling.statsmodels import ContrastInfo
from bids.layout import BIDSLayout
from bids.tests import get_test_data_path
from bids.variables import BIDSVariableCollection


@pytest.fixture
def graph():
    layout_path = join(get_test_data_path(), "ds005")
    layout = BIDSLayout(layout_path)
    json_file = join(layout_path, "models", "ds-005_type-test_model.json")
    graph = BIDSStatsModelsGraph(layout, json_file)
    graph.load_collections(scan_length=480, subject=["01", "02"])
    return graph


def test_first_level_sparse_design_matrix(graph):
    collections = graph["run"].get_collections(subject=["01"])
    assert len(collections) == 3
    df = collections[0].to_df(format="long")
    assert df.shape == (172, 9)
    assert df["condition"].nunique() == 2
    assert set(df.columns) == {
        "amplitude",
        "onset",
        "duration",
        "condition",
        "subject",
        "run",
        "task",
        "datatype",
        "suffix",
    }


def test_incremental_data_loading():
    layout_path = join(get_test_data_path(), "ds005")
    layout = BIDSLayout(layout_path)
    json_file = join(layout_path, "models", "ds-005_type-test_model.json")
    graph = BIDSStatsModelsGraph(layout, json_file)
    graph.load_collections(scan_length=480, subject=["01"], run=[1])
    graph.load_collections(scan_length=480, subject=["02"], run=[2])
    assert len(graph["run"].get_collections()) == 2


def test_post_first_level_sparse_design_matrix(graph):

    collections = graph["session"].get_collections()
    assert len(collections) == 2
    result = collections[0].to_df(format="long")
    assert result.shape == (9, 11)
    result = collections[0].to_df(format="long", entities=False)
    assert result.shape == (9, 2)
    entities = {
        # 'subject': '01',  # PY35
        "task": "mixedgamblestask",
        "datatype": "func",
        "suffix": "bold",
    }
    assert not set(entities.keys()) - set(collections[0].entities.keys())
    assert not set(entities.values()) - set(collections[0].entities.values())
    # PY35
    assert "subject" in collections[0].entities
    assert collections[0].entities["subject"] in ("01", "02")

    # Participant level and also check integer-based indexing
    collections = graph["participant"].get_collections()
    assert len(collections) == 2
    assert graph[2].name == "participant"

    # Dataset level
    collections = graph["group"].get_collections()
    assert len(collections) == 1
    data = collections[0].to_df(format="wide")
    assert len(data) == 2
    assert data["subject"].nunique() == 2

    # # Make sure columns from different levels exist
    varset = {"age", "RT-trial_type", "RT", "crummy-F"}
    assert not varset - set(data.columns.tolist())

    # Calling an invalid level name should raise an exception
    with pytest.raises(KeyError):
        result = graph["nonexistent_name"].to_df()


def test_step_get_collections(graph):
    collections = graph["run"].get_collections(subject="01")
    assert len(collections) == 3
    assert isinstance(collections[0], BIDSVariableCollection)


def test_contrast_info(graph):
    colls = graph["run"].get_collections(subject="01")
    contrast_lists = [graph["run"].get_contrasts(c) for c in colls]
    assert len(contrast_lists) == 3
    for cl in contrast_lists:
        assert len(cl) == 3
        cl = [c for c in cl if c.type == "t"]
        assert set([c.name for c in cl]) == {"RT", "RT-trial_type"}
        assert set([c.type for c in cl]) == {"t"}
        assert cl[0].weights.columns.tolist() == ["RT", "trial_type"]
        assert cl[1].weights.columns.tolist() == ["RT"]
        assert np.array_equal(cl[0].weights.values, np.array([[1, -1]]))
        assert np.array_equal(cl[1].weights.values, np.array([[1]]))
        assert isinstance(cl[0], ContrastInfo)
        assert cl[0]._fields == ("name", "weights", "type", "entities")


def test_contrast_info_with_specified_variables(graph):
    varlist = ["RT", "dummy"]
    colls = graph["run"].get_collections(subject="01")
    contrast_lists = [
        graph["run"].get_contrasts(c, variables=varlist) for c in colls
    ]
    assert len(contrast_lists) == 3
    for cl in contrast_lists:
        assert len(cl) == 3
        cl = [c for c in cl if c.type == "t"]
        assert set([c.name for c in cl]) == {"RT", "RT-trial_type"}
        assert set([c.type for c in cl]) == {"t"}
        for c in cl:
            assert c.weights.columns.tolist() == ["RT", "dummy"]
            assert np.array_equal(c.weights.values, np.array([[1, 0]]))
        assert isinstance(cl[0], ContrastInfo)
        assert cl[0]._fields == ("name", "weights", "type", "entities")


def test_contrast_info_F_contrast(graph):
    colls = graph["run"].get_collections(subject="01")
    contrast_lists = [
        graph["run"].get_contrasts(c, names=["crummy-F"]) for c in colls
    ]
    assert len(contrast_lists) == 3
    for cl in contrast_lists:
        assert len(cl) == 1
        c = cl[0]
        assert c.name == "crummy-F"
        assert c.type == "F"
        assert c.weights.columns.tolist() == ["RT", "trial_type"]
        assert np.array_equal(c.weights.values, np.array([[1, 0], [0, 1]]))
        assert isinstance(c, ContrastInfo)
        assert c._fields == ("name", "weights", "type", "entities")


def test_dummy_contrasts(graph):
    collection = graph["run"].get_collections(subject="01")[0]
    names = [c.name for c in graph["run"].get_contrasts(collection)]

    collection = graph["session"].get_collections(subject="01")[0]
    session = graph["session"].get_contrasts(collection)
    for cl in session:
        assert cl.type == "FEMA"
        assert cl.name in names

    collection = graph["participant"].get_collections(subject="01")[0]
    participant = graph["participant"].get_contrasts(collection)
    assert len(participant) == 3
    for cl in participant:
        assert cl.type == "FEMA"
        assert cl.name in names

    collection = graph["group"].get_collections()[0]
    group = graph["group"].get_contrasts(collection)
    group_names = []
    for cl in group:
        assert cl.type == "t"
        group_names.append(cl.name)

    assert set(names) < set(group_names)


def test_get_run_level_model_spec(graph):
    outputs = graph["run"].run(subject="01", run=1)
    assert len(outputs) == 1
    model_spec = outputs[0].model_spec
    assert model_spec.__class__.__name__ == "GLMMSpec"
    # Note: this implicitly also tests use of formulas, because one is defined
    # in the model for the interaction of RT and gain.
    assert model_spec.X.shape == (240, 4)
    assert model_spec.Z is None
    assert {'RT', 'gain', 'Intercept', 'RT:gain'} == set(model_spec.terms.keys())


def test_entire_graph_smoketest(graph):
    # Smoke test of entire graph; should hit almost all major pieces.
    # We do the following:
    # At run level, construct a design matrix containing gain, RT, and gain * RT.
    # At subject level, aggregate within runs/sessions, do nothing else.
    # At dataset level, do one-sample t-tests separately for each gender,
    # but also two-sample t-tests comparing males and females.
    # Note that there are only 2 subjects in the graph.
    outputs = graph["run"].run(groupby=['subject', 'run'])
    # 2 subjects x 3 runs
    assert len(outputs) == 6
    cis = list(chain(*[op.contrasts for op in outputs]))
    assert len(cis) == 18
    outputs = graph["participant"].run(cis, groupby=['subject', 'contrast'])
    # 2 subjects x 3 contrasts
    assert len(outputs) == 6
    cis = list(chain(*[op.contrasts for op in outputs]))
    assert len(cis) == 6

    # Construct new ContrastInfo objects with name updated to reflect last
    # contrast. This would normally be done by the handling tool (e.g., fitlins)
    inputs = []
    for op in outputs:
        fields = dict(op.contrasts[0]._asdict())
        contrast_name = op.metadata['contrast'].iloc[0]
        fields['name'] = contrast_name
        fields['entities']['contrast'] = contrast_name
        inputs.append(ContrastInfo(**fields))

    # GROUP DIFFERENCE NODE
    outputs = graph["group-diff"].run(inputs, groupby=['contrast'])
    # 3 contrasts
    assert len(outputs) == 3
    cis = list(chain(*[op.contrasts for op in outputs]))
    # 3 contrasts x 2 subjects
    assert len(cis) == 6
    model_spec = outputs[0].model_spec
    assert model_spec.__class__.__name__ == "GLMMSpec"
    assert model_spec.X.shape == (2, 2)
    assert model_spec.Z is None
    assert {"Intercept", "sex"} == set(model_spec.terms.keys())

    # BY-GROUP NODE
    outputs = graph["by-group"].run(inputs, groupby=['contrast'])
    # 3 contrasts
    assert len(outputs) == 3
    cis = list(chain(*[op.contrasts for op in outputs]))
    # two groups x 3 contrasts
    assert len(cis) == 3
    model_spec = outputs[0].model_spec
    assert model_spec.__class__.__name__ == "GLMMSpec"
    assert model_spec.X.shape == (2, 1)
    assert model_spec.Z is None
    assert {"Intercept"} == set(model_spec.terms.keys())
