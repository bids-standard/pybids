# layout
Use grabbit to interact with BIDS projects

## Overview

[Grabbit](https://github.com/grabbles/grabbit) is a lightweight Python 2 and 3 package for simple queries over filenames within a project. It's geared towards projects or applications with highly structured filenames that allow useful queries to be performed without having to inspect the file metadata or contents. Layout is a BIDS-specific extension of grabbit that makes it particularly easy to work with BIDS projects, and provides additional functionality.

## Quickstart

Suppose we have a BIDS project directory that looks like this (partial listing):

```
├── dataset_description.json
├── participants.tsv
├── sub-01
│   ├── ses-1
│   │   ├── anat
│   │   │   ├── sub-01_ses-1_T1map.nii.gz
│   │   │   └── sub-01_ses-1_T1w.nii.gz
│   │   ├── fmap
│   │   │   ├── sub-01_ses-1_run-1_magnitude1.nii.gz
│   │   │   ├── sub-01_ses-1_run-1_magnitude2.nii.gz
│   │   │   ├── sub-01_ses-1_run-1_phasediff.json
│   │   │   ├── sub-01_ses-1_run-1_phasediff.nii.gz
│   │   │   ├── sub-01_ses-1_run-2_magnitude1.nii.gz
│   │   │   ├── sub-01_ses-1_run-2_magnitude2.nii.gz
│   │   │   ├── sub-01_ses-1_run-2_phasediff.json
│   │   │   └── sub-01_ses-1_run-2_phasediff.nii.gz
│   │   ├── func
│   │   │   ├── sub-01_ses-1_task-rest_acq-fullbrain_run-1_bold.nii.gz
│   │   │   ├── sub-01_ses-1_task-rest_acq-fullbrain_run-1_physio.tsv.gz
│   │   │   ├── sub-01_ses-1_task-rest_acq-fullbrain_run-2_bold.nii.gz
│   │   │   ├── sub-01_ses-1_task-rest_acq-fullbrain_run-2_physio.tsv.gz
│   │   │   ├── sub-01_ses-1_task-rest_acq-prefrontal_bold.nii.gz
│   │   │   └── sub-01_ses-1_task-rest_acq-prefrontal_physio.tsv.gz
│   │   └── sub-01_ses-1_scans.tsv
│   ├── ses-2
│   │   ├── fmap
│   │   │   ├── sub-01_ses-2_run-1_magnitude1.nii.gz
│   │   │   ├── sub-01_ses-2_run-1_magnitude2.nii.gz
│   │   │   ├── sub-01_ses-2_run-1_phasediff.json
│   │   │   ├── sub-01_ses-2_run-1_phasediff.nii.gz
│   │   │   ├── sub-01_ses-2_run-2_magnitude1.nii.gz
│   │   │   ├── sub-01_ses-2_run-2_magnitude2.nii.gz
│   │   │   ├── sub-01_ses-2_run-2_phasediff.json
│   │   │   └── sub-01_ses-2_run-2_phasediff.nii.gz
```

We can initialize a `layout` Layout object like so:

```python
from bids.layout import BIDSLayout
project_root = '/my_bids_project'
layout = BIDSLayout(project_root)
```

The `BIDSLayout` instance is a lightweight container for all of the files in the BIDS project directory. It automatically detects any BIDS entities found in the file paths, and allows us to perform simple but relatively powerful queries over the file tree. By default, defined BIDS entities include things like "subject", "session", "run", and "type". In case you're curious, the definitions in the config file look like this (though you probably won't ever have to define them yourself):

```json
{
  "name": "subject",
  "pattern": "sub-([a-zA-Z0-9]+)",
  "directory": "{subject}"
},
{
  "name": "session",
  "pattern": "ses-([a-zA-Z0-9]+)",
  "mandatory": false,
  "directory": "{subject}/{session}",
  "missing_value": "ses-1"
},
{
  "name": "run",
  "pattern": "run-(\\d+)"
},
{
  "name": "type",
  "pattern": "sub-[a-zA-Z0-9_-]+_(.*?)\\."
},
```

### Getting unique values and counts
Once we've initialized a `Layout`, we can do simple things like getting a list of subject labels

```python
>>> layout.get_subjects()
['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
```

### Querying and filtering
Counting is kind of trivial; everyone can count! More usefully, we can run simple logical queries, returning the results in a variety of formats:

```python
>>> files = layout.get(subject='0[12]', run=1, extensions='.nii.gz')
>>> files[0]
File(filename='7t_trt/sub-02/ses-1/fmap/sub-02_ses-1_run-1_magnitude1.nii.gz', subject='sub-02', run='run-1', session='ses-1', type='magnitude1')

>>> [f.path for f in files]
['7t_trt/sub-02/ses-2/fmap/sub-02_ses-2_run-1_phasediff.nii.gz',
 '7t_trt/sub-01/ses-2/func/sub-01_ses-2_task-rest_acq-fullbrain_run-1_bold.nii.gz',
 '7t_trt/sub-02/ses-1/fmap/sub-02_ses-1_run-1_phasediff.nii.gz',
 ...,
 ]
```
In the above snippet, we retrieve all files with subject id 1 or 2 and run id 1 (notice that any entity defined in the config file can be used a filtering argument), and with a file extension of .nii.gz. The returned result is a list of named tuples, one per file, allowing direct access to the defined entities as attributes.

Some other examples of `get()` requests:

```python
>>> # Return all unique 'session' directories
>>> layout.get(target='session', return_type='dir')
['7t_trt/sub-08/ses-1',
 '7t_trt/sub-06/ses-2',
 '7t_trt/sub-01/ses-2',
 ...
 ]

>>> # Return a list of unique file types available for subject 1
>>> layout.get(target='type', return_type='id', subject=1)
['T1map', 'magnitude2', 'magnitude1', 'scans', 'bold', 'phasediff', 'T1w', 'physio']
```

### Get all metadata for a given file

```python
>>> layout.get_metadata('sub-03/ses-2/func/sub-03_ses-2_task-rest_acq-fullbrain_run-2_bold.nii.gz')["RepetitionTime"]
3.0
```

### Get a fieldmap files intended for a given map

```python
>>> layout.get_fieldmap('7t_trt/sub-03/ses-2/func/sub-03_ses-2_task-rest_acq-fullbrain_run-2_bold.nii.gz')
{
  'type': 'phasediff',
  'phasediff': 'sub-03/ses-2/fmap/sub-03_ses-2_run-1_phasediff.nii.gz',
  'magnitude1': 'sub-03/ses-2/fmap/sub-03_ses-2_run-1_magnitude1.nii.gz',
  'magnitude2': 'sub-03/ses-2/fmap/sub-03_ses-2_run-1_magnitude2.nii.gz'
}

```

### For DIYers
If you want to run more complex queries, `layout` provides an easy way to return the full project tree (or a subset of it) as a pandas DataFrame:

```python
>>> # Return all session 1 files as a pandas DF
>>> layout.to_df(session=1)
```

Each row is a single file, and each defined entity is automatically mapped to a column in the DataFrame.
