---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Writing methods reports with `pybids.reports`

This tutorial 

```{code-cell} ipython3
%load_ext autoreload
%autoreload 2

from bids import BIDSLayout
from bids.reports import BIDSReport
from bids.tests import get_test_data_path
import os
```

```{code-cell} ipython3
dataset = "synthetic"

# Here we're using an example BIDS dataset that's bundled with the pybids tests
data_path = os.path.join(get_test_data_path(), dataset)

# Load the BIDS dataset
layout = BIDSLayout(data_path)
```

```{code-cell} ipython3
# Initialize a report for the dataset
report = BIDSReport(layout)
```

```{code-cell} ipython3
# Method generate returns a Counter of unique descriptions across subjects
descriptions = report.generate()
```

```{code-cell} ipython3
# For datasets containing a single study design, all but the most common
# description most likely reflect random missing data.
pub_description = descriptions.most_common()[0][0]
print(pub_description)
```
