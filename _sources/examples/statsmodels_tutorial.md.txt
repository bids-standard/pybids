---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Executing and inspecting StatsModels with `pybids.modeling`
A minimalistic tutorial illustrating usage of the tools in the `bids.modeling` module—most notably, `BIDSStatsModelsGraph` and its various components.

```{code-cell} ipython3
%load_ext autoreload
%autoreload 2

import pandas as pd
import numpy as np
from bids.modeling import BIDSStatsModelsGraph
from bids.layout import BIDSLayout
from bids.tests import get_test_data_path
from os.path import join
import json
```

## Load BIDSLayout
Standard stuff: load the `BIDSLayout` object (we'll use the ds005 dataset bundled with PyBIDS tests) and read in one of the included model files (`ds005/models/ds-005_type-test_model.json`).

```{code-cell} ipython3
layout_path = join(get_test_data_path(), 'ds005')
layout = BIDSLayout(layout_path)
json_file = join(layout_path, 'models', 'ds-005_type-test_model.json')
spec = json.load(open(json_file, 'r'))
```

## Initialize the graph
The `BIDSStatsModelsGraph` is the primary interface to design data, model constructions, etc. We build it from a `BIDSLayout` and the contents of a JSON model file.

```{code-cell} ipython3
graph = BIDSStatsModelsGraph(layout, spec)
```

## Load variables from BIDS dataset
We will typically want to load variables into BIDS "collections" from the BIDS project. The `scan_length` argument is only necessary here because the test dataset doesn't actually include nifti image headers, so duration of scans can't be inferred. In typical use, you can call this without arguments.

```{code-cell} ipython3
graph.load_collections(scan_length=480)
```

## Access specific nodes
Now that the graph is loaded, we can start interacting with its nodes. We'll typically start with the root node, which will usually contain the run-level model information. We can either access the `.root_node`, or use `get_node()` to get the node by its unique name (defined in the JSON file).

```{code-cell} ipython3
# Equivalent to calling graph.get_node('run') in this case.
root_node = graph.root_node
```

## Inspect `BIDSVariableCollection`

+++

We can take a look at the original variables available for each node prior to running the node (i.e. applying any transformations)

```{code-cell} ipython3
colls = root_node.get_collections()
first_sub = colls[0] # Collection for the first subject / run 
```

```{code-cell} ipython3
first_sub.to_df(entities=False)
```

Note that by default `to_df` combines both sparse and dense variables. 

We can take a look at the individual variable objects as well:

```{code-cell} ipython3
first_sub.variables
```

## Executing a node
We can "run" each node to produce design matrices for each unique combination of entities/inputs we want. This is determined by the grouping structure. When working with the API directly, this is indicated via the `group_by` argument to a `.run()` method. In this case, we're indicating that the design information should be set up separately for every unique combination of `run` and `subject`.

Note that we need to include `subject` even though this is a strictly run-level model, because if we only pass `['run']`, there will only be 3 separate analyses conducted: one for run 1, one for run 2, and one for run 3. Whereas what we actually want is for the procedure to be done separately for each unique combination of the 3 runs and 16 subjects (i.e., 48 times).

```{code-cell} ipython3
# force_dense controls whether the output for run-level design matrices
# will be resampled to a uniform "dense" representation, or left alone
# as a sparse representation if possible.
outputs = root_node.run(group_by=['run', 'subject'], force_dense=False, transformation_history=True)
```

## Node outputs

+++

The result is a list of objects of type `BIDSStatsModelsNodeOutput`. This is a lightweight container that stores design information and various other useful pieces of information. There should one element in the list for each unique combination of the grouping variables (in this case, run and subject):

```{code-cell} ipython3
len(outputs)
```

```{code-cell} ipython3
# The first 10 `BIDSStatsModelsNodeOutput` objects
outputs[:10]
```

Let's take a closer look at the `BIDSStatsModelsNodeOutput`. First, we have an `.entities` attribute that tells us what levels of the grouping variables this output corresponds to:

```{code-cell} ipython3
outputs[0].entities
```

Next, we can get the design matrix itself via the `.X` attribute:

```{code-cell} ipython3
outputs[0].X
```

Here we have a column for each of the contrasts specified in the model (with the same name as the contrast). We can access the contrasts too, via—you guessed it—`.contrasts`:

```{code-cell} ipython3
outputs[0].contrasts
```

Each `ContrastInfo` is a named tuple with fields that map directly on the definition of contrasts in the BIDS-StatsModels specification. The only addition is the inclusion of an `.entities` field that stores a dictionary of the entities that identify what subset of our data the contrast applies to.

One thing you might be puzzled by, looking at the output of the `.X` call above, is the absence of any timing information. `.X` is supposed to give us a design matrix, but how come the output only has the actual values for each column, and no information about event timing? How are we supposed to know what the onsets, durations, etc. of each event are?

The answer is that `.X` contains *only* the actual values that go into the design matrix, and not any metadata—no matter how important. Fortunately, that metadata is available to us. It's conveniently stored in a `.metadata` attribute on the `BIDSStatsModelsNodeOutput` object.

```{code-cell} ipython3
outputs[0].metadata
```

There's a 1-to-1 mapping from rows in `.X` to rows in `.metadata`. This means you can, if you like, simply concatenate the two along the column axis to get one big DataFrame with everything. But by maintaining a default separation, it's made very clear to us which columns are properly a part of the design, and which contain additional metadata.

+++

## Transformation History

+++

To generate the final design matrix, pybids applies all transformations specified in the model for that given run.

However, it's often useful to look at the intermediary outputs from each transformation to perform sanity checks, such as previewing the variable prior to convolution.

Optionally, you can run the `Node` with `transformation_history=True`, and the `BIDSStatsModelsNodeOutput` object will have a `trans_hist` attribute which is a list of intermediary outputs after every transformation

```{code-cell} ipython3
ts = outputs[0].trans_hist
ts
```

The first item (index=0), is the original collection. We can access the `BIDSRunVariableCollection` using the output attribute:

```{code-cell} ipython3
ts[0].output
```

We can examine the full collection at each step using either a combined dataframe:

```{code-cell} ipython3
ts[1].output.to_df(entities=False)
```

Or examine individual variables in their native representation (e.g. `SparseRunVariable` or `DenseRunVariable`)

```{code-cell} ipython3
ts[1].output.variables
```

```{code-cell} ipython3
ts[1].output.variables['RT'].to_df(entities=False)
```

## Traversing the graph
So far we've executed the root node, which by definition required no inputs from any previous node. But in typical workflows, we'll be passing outputs from one node in as inputs to another. For example, we often want to take the run-level parameter estimates and pass them to a subject-level model that does nothing but average over runs within each subject. This requires us to somehow traverse the graph based on the edges specified in the BIDS-StatsModel document. We can do that by taking advantage of each node's `.children` attribute, which contains a list of `BIDSStatsModelsEdge` named tuples that specify an edge between two nodes.

```{code-cell} ipython3
root_node.children
```

In this case the root node connects to only one other node. We can directly access that node by following the `.destination` property of the edge:

```{code-cell} ipython3
next_node = root_node.children[0].destination
next_node.level, next_node.name
```

We assign the first connected node to `next_node`, and print out its `level` and `name` for inspection (both are session).

## Passing in inputs
Armed with that, we can run the session node and proceed and before. However, there's a twist: whereas the root node only needs to know about variables loaded directly from the `BIDSLayout` (which we achieved with that `.load_collections()` call earlier), the session node can't get the inputs it needs from the `BIDSLayout`, because there aren't any (at least in this particular dataset). What we want to do at the session level is average over our run-level estimates within-session. But to do that, we need to actually pass in information about those runs!

The way we do this is to pass, as the first argument to `.run()`, a list of `ContrastInfo` objects informing our node about what inputs it should use to construct its design matrices. The typical use pattern is to pass one concatenated list containing *all* of the outputs from the previous level that we want to pass on. Note that we may not want to pass *all* of the outputs forward. For example, suppose that 2 out of 48 run-level models failed during the estimation process. We might not want to keep passing information about those 2 runs forward, as we can't compute them. So we can always filter the list of `ContrastInfo` objects we received from the previous node before we pass them on to the next node. (We could also do other things, like rename each `ContrastInfo` to use whatever naming scheme our software prefers; modifying the entities; and so on. But we won't do any of that here.)

Let's concatenate the 48 outputs we got from the previous level and drop the last 2, in preparation for passing them forward to our `next_node`:

```{code-cell} ipython3
from itertools import chain
contrasts = list(chain(*[s.contrasts for s in outputs[:-2]]))
len(contrasts)
```

Notice that we're left we're 138 individual `ContrastInfo` objects. Why 138? Because we have ((3 runs x 16 subjects) - 2) * 3 contrasts. Recall that we're dropping the last two of the 48 outputs from the previous level. But each of those lists contains *three* `ContrastInfo` objects (one per contrast—`RT`, `gain`, and the `RT:gain` interaction). Hence, 138.

Now we can call `.run()` on our session-level node, passing in the contrasts as inputs. We want the model specification (i.e., the part in the `"Model"` section of the node) to be applied separately to each unique combination of `contrast`, `session`, and `subject`.

```{code-cell} ipython3
sess_outputs = next_node.run(contrasts, group_by=['subject', 'contrast'])
len(sess_outputs)
```

Again we get back a list of `BIDSStatsModelsNodeOutput` objects. And again we have 48 of them. It might seem odd that we have the same number of outputs from a subject-level node as we had from the run-level node, but there's actually a difference. In the run-level case, our 48 results reflected 16 subjects x 3 runs. In the subject-level case, we *have* successfully aggregated over runs within each subject, but we now have 3 sets of contrasts producing outputs (i.e., 16 subjects x 3 contrasts).

This becomes clearer if we inspect the same attributes we looked at earlier:

```{code-cell} ipython3
sess_outputs[0].contrasts
```

```{code-cell} ipython3
# Concatenate the X and metadata DFs for easy reading.
# Note that only the first column is actually part of the
# design matrix; the others are just metadata.
pd.concat([sess_outputs[0].X, sess_outputs[0].metadata], axis=1)
```

Notice how the entities differ: the run-level node grouped on `run` and `subject`; the subject-level node groups on `subject` and `contrast`. The number of outputs is identical in both cases, but this is just an (un)happy accident, not a general principle. You can verify this for yourself by re-running the subject-level node with a different grouping (e.g., only `['subject']`).
