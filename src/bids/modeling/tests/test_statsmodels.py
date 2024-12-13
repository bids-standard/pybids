from os.path import join
from os import path
from itertools import chain

from nibabel.optpkg import optional_package
graphviz, has_graphviz, _ = optional_package("graphviz")

import pytest
import numpy as np 
import json 

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

@pytest.fixture
def graph_intercept():
    layout_path = join(get_test_data_path(), "ds005")
    layout = BIDSLayout(layout_path, derivatives=join(layout_path, 'derivatives', 'events'))
    json_file = join(layout_path, "models", "ds-005_type-test_intercept.json")
    graph = BIDSStatsModelsGraph(layout, json_file)
    graph.load_collections(scan_length=480, subject=["01", "02"])
    return graph

@pytest.fixture
def graph_nodummy():
    layout_path = join(get_test_data_path(), "ds005")
    layout = BIDSLayout(layout_path)
    json_file = join(layout_path, "models", "ds-005_type-testnodummy_model.json")
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

def test_repr(graph):
    assert graph.__repr__() == "<BIDSStatsModelsGraph[{name='test_model', description='simple test model', ... }]>"
    node = graph.nodes['run']
    assert node.__repr__() == "<BIDSStatsModelsNode(level=run, name=run)>"
    assert node.run()[0].__repr__() == "<BIDSStatsModelsNodeOutput(name=run, entities={'run': 1, 'subject': '01'})>"

def test_manual_intercept(graph_intercept):
    # Test that a automatic intercept (1) is correct
    # Intercept could should be all 1s
    run = graph_intercept["run"]
    outputs = run.run(subject="01", run=1)
    assert outputs[0].X.intercept.min() == 1.0

    # Defining both 1 and 'intercept' raises an error
    run.model['x'] = [1, 'intercept']
    with pytest.raises(ValueError, match="Cannot define both '1' and 'intercept' in 'X'"):
        run.run(subject="01", run=1)

    # "intercept" variable from event files is loaded correctly (should not be all 1s)
    run = graph_intercept.nodes['run']
    run.model['x'] = ['intercept']
    outputs= run.run(subject="01", run=1)
    assert outputs[0].X.intercept.min() != 1.0


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


def test_contrast_dummy_vs_explicit(graph, graph_nodummy):
    # Check that ContrastInfo from model w/ DummyContrasts
    # and explicit Contrasts are identical
    outputs = graph["run"].run(subject="01", run=1)
    outputs_nodummy = graph_nodummy["run"].run(subject="01", run=1)

    for con in outputs[0].contrasts:
        match = [c for c in outputs_nodummy[0].contrasts if c.name == con.name][0]

        assert con.conditions == match.conditions
        assert con.weights == match.weights
        assert con.test == match.test

    cis = list(chain(*[op.contrasts for op in outputs]))
    outputs_sub = graph["participant"].run(cis, group_by=['subject', 'contrast'])
    output = [o for o in outputs_sub if o.entities['contrast'] == 'RT'][0]

    cis = list(chain(*[op.contrasts for op in outputs_nodummy]))
    outputs_nodummy_sub = graph_nodummy["participant"].run(cis, group_by=['subject', 'contrast'])
    output_nodummy = [o for o in outputs_nodummy_sub if o.entities['contrast'] == 'RT'][0]

    for con in output.contrasts:
        match = [c for c in output_nodummy.contrasts if c.name == con.name][0]

        assert con.conditions == match.conditions
        assert con.weights == match.weights
        assert con.test == match.test

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
    outputs = graph["run"].run()
    # 2 subjects x 3 runs
    assert len(outputs) == 6
    cis = list(chain(*[op.contrasts for op in outputs]))
    assert len(cis) == 18
    outputs = graph["participant"].run(cis)
    # 2 subjects x 3 contrasts)
    assert len(outputs) == 6
    # * 2 participant level contrasts = 12
    cis = list(chain(*[op.contrasts for op in outputs]))
    assert len(cis) == 12

    # Test output names for single subject
    out_contrasts = [
        c.entities['contrast'] for c in cis if c.entities['subject'] == '01'
        ]

    expected_outs = [
        'gain', 'gain_neg', 'RT', 'RT_neg', 'RT:gain', 'RT:gain_neg'
    ]

    assert set(out_contrasts) == set(expected_outs)

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
    outputs = graph["group-diff"].run(inputs)
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
    assert not set(model_spec.terms.keys()) - {"intercept", "sex"}

    # BY-GROUP NODE
    outputs = graph["by-group"].run(inputs)
    # 3 contrasts
    assert len(outputs) == 6
    cis = list(chain(*[op.contrasts for op in outputs]))
    # two groups x 3 contrasts
    assert len(cis) == 6
    model_spec = outputs[0].model_spec
    assert model_spec.__class__.__name__ == "GLMMSpec"
    assert model_spec.X.shape == (1, 1)
    assert model_spec.Z is None
    assert not set(model_spec.terms.keys()) - {"intercept"}

    # explicit-contrast NODE
    outputs = graph["explicit-contrast"].run(inputs)
    # 1 group x 1 contrast
    assert len(outputs) == 1
    assert len(outputs[0].contrasts) == 1
    assert outputs[0].X['gain'].sum() == 2
    model_spec = outputs[0].model_spec
    assert model_spec.__class__.__name__ == "GLMMSpec"
    assert model_spec.X.shape == (6, 1)
    assert not set(model_spec.terms.keys()) - {"gain"}

    contrast = outputs[0].contrasts[0]
    assert contrast.name == 'gain'


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


def test_interceptonly_runlevel_error():
    layout_path = join(get_test_data_path(), "ds005")
    layout = BIDSLayout(layout_path)
    json_file = join(layout_path, "models", "ds-005_type-interceptonlyrunlevel_model.json")
    with pytest.raises(NotImplementedError):
        graph = BIDSStatsModelsGraph(layout, json_file)

def test_missing_value_fill():
    layout_path = join(get_test_data_path(), "ds005")
    layout = BIDSLayout(layout_path, derivatives=join(layout_path, 'derivatives', 'fmriprep'))
    json_file = join(layout_path, "models", "ds-005_type-test_model.json")

    model = json.load(open(json_file))
    # Add 'global_signal_derivative1' to the model
    model['Nodes'][0]['Model'] = {'X': ['RT', 'gain', 'global_signal_derivative1']}

    graph = BIDSStatsModelsGraph(layout, model)
    graph.load_collections(scan_length=480, subject=["01", "02"])

    # Check that missing values are filled in
    # Assert that a warnings is raised
    with pytest.warns(UserWarning):
        graph.run_graph()

    # Assert that there are no missing values
    outputs = graph.nodes['run'].outputs_
    assert not np.isnan(outputs[0].model_spec.X).any().any()

    # Check that missing_values='error' raises an error
    with pytest.raises(ValueError):
        graph.run_graph(missing_values='error')

    # Check that missing_values='ignore' passes through missing values
    graph.run_graph(missing_values='ignore')

    outputs = graph.nodes['run'].outputs_
    assert np.isnan(outputs[0].model_spec.X).any().any()