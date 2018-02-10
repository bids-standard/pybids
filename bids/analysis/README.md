## PyBIDS: Analysis

The PyBIDS analysis module provides high-level model specification functionality for BIDS datasets. It assumes that model information is represented in line with the (upcoming) BIDS-Model specification.

Note that, at present, pybids.analysis does *not* provide a one-stop model-fitting solution. You will need to call some other package (e.g., nistats, or a non-Python package like FSL or SPM via Nipype) to handle model estimation. What pybids.analysis *will* do for you is automatically handle the loading and transformation of all variables, and the construction of design matrices and contrasts.

### Quickstart

A simple example of standard usage follows. We assume that we have a root folder containing a BIDS-compliant project in `/bidsproject`, as well as a BIDS-Models JSON specification in `model.json`.

```python
from bids.analysis import Analysis

# Initialize the Analysis
analysis = Analysis('/bidsproject', 'model1.json')

# Setup constructs all the design matrices
analysis.setup()

# Sample query: retrieve first-level design matrix for one run
dm = analysis[0].get_design_matrix(subject='01', run=1, task='taskA')

# Sample query: retrieve session-level contrast matrix
cm = analysis[1].get_contrasts(subject='01', session='retest')
```
