from os.path import join
from os import path
from itertools import chain

from nibabel.optpkg import optional_package
graphviz, has_graphviz, _ = optional_package("graphviz")

import os
import numpy as np
import pytest

from bids.modeling import BIDSStatsModelsGraph
from bids.modeling.statsmodels import ContrastInfo, expand_wildcards
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

@pytest.mark.skipif(not has_graphviz, reason="Test requires graphviz")
def test_write_graph(graph, tmp_path):
    from graphviz import Digraph

    dot = graph.write_graph(tmp_path / "graph.dot")
    assert isinstance(dot, Digraph)
    assert path.exists(tmp_path / "graph.dot")
    assert path.exists(tmp_path / "graph.dot.png")

def test_first_level_sparse_design_matrix(graph):
    outputs = graph["run"].run(subject=["01"], force_dense=False)
    assert len(outputs) == 3
    df = outputs[0].X
    assert df.shape == (86, 3)
    assert set(df.columns) == {'RT', 'gain', 'RT:gain'}
    metadata = outputs[0].metadata
    assert metadata.shape == (86, 7)
    assert set(metadata.columns) == {
        "onset",
        "duration",
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


def test_step_get_collections(graph):
    collections = graph["run"].get_collections(subject="01")
    assert len(collections) == 3
    assert isinstance(collections[0], BIDSVariableCollection)


def test_contrast_info(graph):
    outputs = graph["run"].run(subject="01")
    contrast_lists = [op.contrasts for op in outputs]
    assert len(contrast_lists) == 3
    for cl in contrast_lists:
        assert len(cl) == 3
        cl = [c for c in cl if c.test == "t"]
        assert set([c.name for c in cl]) == {"RT", "RT:gain", "gain"}
        assert set([c.test for c in cl]) == {"t"}
        assert len(cl[1].conditions) == 1
        assert cl[1].conditions[0] in {'RT', 'gain', 'RT:gain'}
        assert cl[1].weights == [1]
        assert isinstance(cl[0], ContrastInfo)
        assert cl[0]._fields == ("name", "conditions", "weights", "test", "entities")


def test_get_run_level_model_spec(graph):
    outputs = graph["run"].run(subject="01", run=1)
    assert len(outputs) == 1
    model_spec = outputs[0].model_spec
    assert model_spec.__class__.__name__ == "GLMMSpec"
    # Note: this implicitly also tests use of formulas, because one is defined
    # in the model for the interaction of RT and gain.
    assert model_spec.X.shape == (240, 3)
    assert model_spec.Z is None
    assert {'RT', 'gain', 'RT:gain'} == set(model_spec.terms.keys())


def test_entire_graph_smoketest(graph):
    # Smoke test of entire graph; should hit almost all major pieces.
    # We do the following:
    # At run level, construct a design matrix containing gain, RT, and gain * RT.
    # At subject level, aggregate within runs, do nothing else.
    # At dataset level, do one-sample t-tests separately for each gender,
    # but also two-sample t-tests comparing males and females.
    # Note that there are only 2 subjects in the graph.
    # Note also that there is only one session (with no session label), which
    # should have no effect as a grouping variable
    outputs = graph["run"].run(group_by=['subject', 'session', 'run'])
    # 2 subjects x 3 runs
    assert len(outputs) == 6
    cis = list(chain(*[op.contrasts for op in outputs]))
    assert len(cis) == 18
    outputs = graph["participant"].run(cis, group_by=['subject', 'contrast'])
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
    outputs = graph["group-diff"].run(inputs, group_by=['contrast'])
    # 3 contrasts
    assert len(outputs) == 3
    cis = list(chain(*[op.contrasts for op in outputs]))
    # 3 contrasts x 2 subjects
    assert len(cis) == 6
    model_spec = outputs[0].model_spec
    assert model_spec.__class__.__name__ == "GLMMSpec"
    assert model_spec.X.shape == (2, 2)
    assert model_spec.Z is None
    assert len(model_spec.terms) == 2
    assert not set(model_spec.terms.keys()) - {"RT", "gain", "RT:gain", "sex"}

    # BY-GROUP NODE
    outputs = graph["by-group"].run(inputs, group_by=['contrast'])
    # 3 contrasts
    assert len(outputs) == 3
    cis = list(chain(*[op.contrasts for op in outputs]))
    # two groups x 3 contrasts
    assert len(cis) == 3
    model_spec = outputs[0].model_spec
    assert model_spec.__class__.__name__ == "GLMMSpec"
    assert model_spec.X.shape == (2, 1)
    assert model_spec.Z is None
    assert not set(model_spec.terms.keys()) - {"RT", "gain", "RT:gain"}


def test_expand_wildcards():
    # No wildcards == no modification
    assert expand_wildcards(["a", "b"], ["a", "c"]) == ["a", "b"]
    # No matches == removal
    assert expand_wildcards(["a", "b*"], ["a", "c"]) == ["a"]
    # Matches expand in-place
    assert expand_wildcards(["a*", "b"], ["a", "c"]) == ["a", "b"]
    assert expand_wildcards(["a*", "b"], ["a0", "c", "a1", "a2"]) == ["a0", "a1", "a2", "b"]
    # Some examples
    assert expand_wildcards(
        ["trial_type.*"], ["trial_type.A", "trial_type.B"]
    ) == ["trial_type.A", "trial_type.B"]
    assert expand_wildcards(
        ["non_steady_state*"], ["non_steady_state00", "non_steady_state01"]
    ) == ["non_steady_state00", "non_steady_state01"]
