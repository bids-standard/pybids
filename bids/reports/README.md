## PyBIDS: Reports

The PyBIDS reports module generates publication-quality data acquisition descriptions from BIDS datasets.

NOTE: The reports module is experimental and currently under active development, and as such should be used with caution.
Please remember to verify any generated report before putting it to use.

Additionally, only MRI datatypes (func, anat, fmap, and dwi) are currently supported.

### Quickstart

A simple example of standard usage follows. We assume that we have a root folder containing a BIDS-compliant project in `/bidsproject`.

```python
from bids.layout import BIDSLayout
from bids.reports import BIDSReport

# Load the BIDS dataset
layout = BIDSLayout('/bidsproject')

# Initialize a report for the dataset
report = BIDSReport(layout)

# Method generate returns a Counter of unique descriptions across subjects
descriptions = report.generate()

# For datasets containing a single study design, all but the most common
# description most likely reflect random missing data.
pub_description = descriptions.most_common()[0][0]
```
