# grabbids
Get grabby with BIDS projects

## Overview

Grabbit is a lightweight Python 2 and 3 package for simple queries over filenames within a project. It's geared towards projects or applications with highly structured filenames that allow useful queries to be performed without having to inspect the file metadata or contents. Grabbids is a BIDS-specific extension of grabbit that makes it particularly easy to work with BIDS projects, and provides additional functionality.

## Installation

```
$ pip install grabbids
```

Or, if you like to (a) do things the hard way or (b) live on the bleeding edge:

```
$ git clone https://github.com/INCF/grabbids
$ cd grabbids
$ python setup.py develop
```

## Quickstart

Suppose we've already defined (or otherwise obtained) a grabbids JSON configuration file that looks [like this](FIXME). And we have a BIDS project directory that looks like this (partial listing):

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

We can initialize a grabbids Layout object like so:

```python
from grabbids import Layout
project_root = '/my_bids_project' 
layout = Layout(project_root)
```

The `Layout` instance is a lightweight container for all of the files in the BIDS project directory. It automatically detects any BIDS entities found in the file paths, and allows us to perform simple but relatively powerful queries over the file tree. By default, defined BIDS entities include things like "subject", "session", "run", and "type". In case you're curious, the definitions in the config file look like this (though you probaby won't ever have to define them yourself):

```json
{
  "name": "subject",
  "pattern": "(sub-\\d+)",
  "directory": "{{root}}/{subject}",
  "mandatory": true
},
{
  "name": "session",
  "pattern": "(ses-\\d)",
  "directory": "{{root}}/{subject}/{session}",
},
{
  "name": "run",
  "pattern": "(run-\\d+)"
},
{
  "name": "type",
  "pattern": ".*_(.*?)\\."
}
```

### Getting unique values and counts
Once we've initialized a `Layout`, we can do simple things like counting and listing all unique values of a given entity:

```python
>>> layout.unique('subject')
['sub-09', 'sub-05', 'sub-08', 'sub-01', 'sub-02', 'sub-06', 'sub-04', 'sub-03', 'sub-07', 'sub-10']

>>> layout.count('run')
2
```

### Querying and filtering
Counting is kind of trivial; everyone can count! More usefully, we can run simple logical queries, returning the results in a variety of formats:

```python
>>> files = layout.get(subject='sub-0[12]', run=1, extensions='.nii.gz')
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

Some other examples of get() requests:

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

### Find the nearest matching entity
A common use case when working with BIDS projects is to find the nearest entity of a given type that matches a particular input image. For example, one might want to find the .bval or .bvec file that applies to a particular nifti image. The location of such files is underdetermined by the BIDS spec (which seeks to minimize redundancy, so it allows metadata and other files to be placed at varying levels of the project hierarchy). Fortunately, we can try to locate the nearest matching file by walking up the file tree from a given target:

```python
>>> layout.find_match(target='bval', source='7t_trt/sub-03/ses-2/func/sub-03_ses-2_task-rest_acq-fullbrain_run-2_bold.nii.gz')
"7t_trt/sub-03/sub-03-test.bval"
```

### For DIYers
If you want to run more complex queries, grabbids provides an easy way to return the full project tree (or a subset of it) as a pandas DataFrame:

```python
>>> # Return all session 1 files as a pandas DF
>>> layout.as_data_frame(session=1)
```

Each row is a single file, and each defined entity is automatically mapped to a column in the DataFrame.
