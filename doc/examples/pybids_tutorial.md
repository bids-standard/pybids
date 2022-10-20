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

# Introduction to `pybids`

[`pybids`](https://github.com/bids-standard/pybids) is a tool to query, summarize and manipulate data using the BIDS standard. 
In this tutorial we will use a `pybids` test dataset to illustrate some of the functionality of `pybids.layout`

```{code-cell} ipython3
from bids import BIDSLayout
from bids.tests import get_test_data_path
import os
```

## The `BIDSLayout`

At the core of pybids is the `BIDSLayout` object. A `BIDSLayout` is a lightweight Python class that represents a BIDS project file tree and provides a variety of helpful methods for querying and manipulating BIDS files. While the `BIDSLayout` initializer has a large number of arguments you can use to control the way files are indexed and accessed, you will most commonly initialize a `BIDSLayout` by passing in the BIDS dataset root location as a single argument:

```{code-cell} ipython3
# Here we're using an example BIDS dataset that's bundled with the pybids tests
data_path = os.path.join(get_test_data_path(), '7t_trt')

# Initialize the layout
layout = BIDSLayout(data_path)

# Print some basic information about the layout
layout
```

### Querying the `BIDSLayout`
When we initialize a `BIDSLayout`, all of the files and metadata found under the specified root folder are indexed. This can take a few seconds (or, for very large datasets, a minute or two). Once initialization is complete, we can start querying the `BIDSLayout` in various ways. The workhorse method is [`.get()`](https://bids-standard.github.io/pybids/generated/bids.grabbids.BIDSLayout.html#bids.grabbids.BIDSLayout.get). If we call `.get()` with no additional arguments, we get back a list of all the BIDS files in our dataset:

```{code-cell} ipython3
all_files = layout.get()
print("There are {} files in the layout.".format(len(all_files)))
print("\nThe first 10 files are:")
all_files[:10]
```

The returned object is a Python list. By default, each element in the list is a `BIDSFile` object. We discuss the `BIDSFile` object in much more detail below. For now, let's simplify things and work with just filenames:

```{code-cell} ipython3
layout.get(return_type='filename')[:10]
```

This time, we get back only the names of the files.

### Filtering files by entities
The utility of the `BIDSLayout` would be pretty limited if all we could do was retrieve a list of all files in the dataset. Fortunately, the `.get()` method accepts all kinds of arguments that allow us to filter the result set based on specified criteria. In fact, we can pass *any* BIDS-defined keywords (or, as they're called in PyBIDS, *entities*) as constraints. For example, here's how we would retrieve all BOLD runs with `.nii.gz` extensions for subject `'01'`:

```{code-cell} ipython3
# Retrieve filenames of all BOLD runs for subject 01
layout.get(subject='01', extension='nii.gz', suffix='bold', return_type='filename')
```

If you're wondering what entities you can pass in as filtering arguments, the answer is contained in the `.json` configuration files [housed here](https://github.com/bids-standard/pybids/tree/master/bids/layout/config). To save you the trouble, here are a few of the most common entities:

* `suffix`: The part of a BIDS filename just before the extension (e.g., `'bold'`, `'events'`, `'physio'`, etc.).
* `subject`: The subject label
* `session`: The session label
* `run`: The run index
* `task`: The task name

New entities are continually being defined as the spec grows, and in principle (though not always in practice), PyBIDS should be aware of all entities that are defined in the BIDS specification.

### Filtering by metadata
All of the entities listed above are found in the names of BIDS files. But sometimes we want to search for files based not just on their names, but also based on metadata defined (per the BIDS spec) in JSON files. Fortunately for us, when we initialize a `BIDSLayout`, all metadata files associated with BIDS files are automatically indexed. This means we can pass any key that occurs in any JSON file in our project as an argument to `.get()`. We can combine these with any number of core BIDS entities (like `subject`, `run`, etc.).

For example, say we want to retrieve all files where (a) the value of `SamplingFrequency` (a metadata key) is `100`, (b) the `acquisition` type is `'prefrontal'`, and (c) the subject is `'01'` or `'02'`. Here's how we can do that:

```{code-cell} ipython3
# Retrieve all files where SamplingFrequency (a metadata key) = 100
# and acquisition = prefrontal, for the first two subjects
layout.get(subject=['01', '02'], SamplingFrequency=100, acquisition="prefrontal")
```

Notice that we passed a list in for `subject` rather than just a string. This principle applies to all filters: you can always pass in a list instead of a single value, and this will be interpreted as a logical disjunction (i.e., a file must match any one of the provided values).

### Other `return_type` values
While we'll typically want to work with either `BIDSFile` objects or filenames, we can also ask `get()` to return unique values (or ids) of particular entities. For example, say we want to know which subjects have at least one `T1w` file. We can request that information by setting `return_type='id'`. When using this option, we also need to specify a target entity (or metadata keyword) called `target`. This combination tells the `BIDSLayout` to return the unique values for the specified `target` entity. For example, in the next example, we ask for all of the unique subject IDs that have at least one file with a `T1w` suffix:

```{code-cell} ipython3
# Ask get() to return the ids of subjects that have T1w files
layout.get(return_type='id', target='subject', suffix='T1w')
```

If our `target` is a BIDS entity that corresponds to a particular directory in the BIDS spec (e.g., `subject` or `session`) we can also use `return_type='dir'` to get all matching subdirectories:

```{code-cell} ipython3
layout.get(return_type='dir', target='subject')
```

### Other `get()` options
The `.get()` method has a number of other useful arguments that control its behavior. We won't discuss these in detail here, but briefly, here are a couple worth knowing about:
* `regex_search`: If you set this to `True`, string filter argument values will be interpreted as regular expressions.
* `scope`: If your BIDS dataset contains BIDS-derivatives sub-datasets, you can specify the scope (e.g., `derivatives`, or a BIDS-Derivatives pipeline name) of the search space.

+++

## The `BIDSFile`
When you call `.get()` on a `BIDSLayout`, the default returned values are objects of class `BIDSFile`. A `BIDSFile` is a lightweight container for individual files in a BIDS dataset. It provides easy access to a variety of useful attributes and methods. Let's take a closer look. First, let's pick a random file from our existing `layout`.

```{code-cell} ipython3
# Pick the 15th file in the dataset
bf = layout.get()[15]

# Print it
bf
```

Here are some of the attributes and methods available to us in a `BIDSFile` (note that some of these are only available for certain subclasses of `BIDSFile`; e.g., you can't call `get_image()` on a `BIDSFile` that doesn't correspond to an image file!):
* `.path`: The full path of the associated file
* `.filename`: The associated file's filename (without directory)
* `.dirname`: The directory containing the file
* `.get_entities()`: Returns information about entities associated with this `BIDSFile` (optionally including metadata)
* `.get_image()`: Returns the file contents as a nibabel image (only works for image files)
* `.get_df()`: Get file contents as a pandas DataFrame (only works for TSV files)
* `.get_metadata()`: Returns a dictionary of all metadata found in associated JSON files
* `.get_associations()`: Returns a list of all files associated with this one in some way

Let's see some of these in action.

```{code-cell} ipython3
# Print all the entities associated with this file, and their values
bf.get_entities()
```

```{code-cell} ipython3
# Print all the metadata associated with this file
bf.get_metadata()
```

```{code-cell} ipython3
# We can the union of both of the above in one shot like this
bf.get_entities(metadata='all')
```

Here are all the files associated with our target file in some way. Notice how we get back both the JSON sidecar for our target file, and the BOLD run that our target file contains physiological recordings for.

```{code-cell} ipython3
bf.get_associations()
```

+++ {"slideshow": {"slide_type": "slide"}}

In cases where a file has a `.tsv.gz` or `.tsv` extension, it will automatically be created as a `BIDSDataFile`, and we can easily grab the contents as a pandas `DataFrame`:

```{code-cell} ipython3
# Use a different test dataset--one that contains physio recording files
data_path = os.path.join(get_test_data_path(), 'synthetic')
layout2 = BIDSLayout(data_path)

# Get the first physiological recording file
recfile = layout2.get(suffix='physio')[0]

# Get contents as a DataFrame and show the first few rows
df = recfile.get_df()
df.head()
```

While it would have been easy enough to read the contents of the file ourselves with pandas' `read_csv()` method, notice that in the above example, `get_df()` saved us the trouble of having to read the physiological recording file's metadata, pull out the column names and sampling rate, and add timing information.

Mind you, if we don't *want* the timing information, we can ignore it:

```{code-cell} ipython3
recfile.get_df(include_timing=False).head()
```

## Other utilities

### Filename parsing
Say you have a filename, and you want to manually extract BIDS entities from it. The `parse_file_entities` method provides the facility:

```{code-cell} ipython3
path = "/a/fake/path/to/a/BIDS/file/sub-01_run-1_T2w.nii.gz"
layout.parse_file_entities(path)
```

A version of this utility independent of a specific layout is available at `bids.layout` ([doc](https://bids-standard.github.io/pybids/generated/bids.layout.parse_file_entities.html#bids.layout.parse_file_entities)) - 

```{code-cell} ipython3
from bids.layout import parse_file_entities

path = "/a/fake/path/to/a/BIDS/file/sub-01_run-1_T2w.nii.gz"
parse_file_entities(path)
```

### Path construction
You may want to create valid BIDS filenames for files that are new or hypothetical that would sit within your BIDS project. This is useful when you know what entity values you need to write out to, but don't want to deal with looking up the precise BIDS file-naming syntax. In the example below, imagine we've created a new file containing stimulus presentation information, and we want to save it to a `.tsv.gz` file, per the BIDS naming conventions. All we need to do is define a dictionary with the name components, and `build_path` takes care of the rest (including injecting sub-directories!):

```{code-cell} ipython3
entities = {
    'subject': '01',
    'run': 2,
    'task': 'nback',
    'suffix': 'bold'
}

layout.build_path(entities)
```

You can also use `build_path` in more sophisticated ways—for example, by defining your own set of matching templates that cover cases not supported by BIDS out of the box. For example, suppose you want to create a template for naming a new z-stat file. You could do something like:

```{code-cell} ipython3
# Define the pattern to build out of the components passed in the dictionary
pattern = "sub-{subject}[_ses-{session}]_task-{task}[_acq-{acquisition}][_rec-{reconstruction}][_run-{run}][_echo-{echo}]_{suffix<z>}.nii.gz",

entities = {
    'subject': '01',
    'run': 2,
    'task': 'nback',
    'suffix': 'z'
}

# Notice we pass the new pattern as the second argument
layout.build_path(entities, pattern, validate=False)
```

Note that in the above example, we set `validate=False` to ensure that the standard BIDS file validator doesn't run (because the pattern we defined isn't actually compliant with the BIDS specification).

+++

### Loading derivatives

By default, `BIDSLayout` objects are initialized without scanning contained `derivatives/` directories. But you can easily ensure that all derivatives files are loaded and endowed with the extra structure specified in the [derivatives config file](https://github.com/bids-standard/pybids/blob/master/bids/layout/config/derivatives.json):

```{code-cell} ipython3
# Define paths to root and derivatives folders
root = os.path.join(get_test_data_path(), 'synthetic')
layout2 = BIDSLayout(root, derivatives=True)
layout2
```

The `scope` argument to `get()` specifies which part of the project to look in. By default, valid values are `'bids'` (for the "raw" BIDS project that excludes derivatives) and `'derivatives'` (for all BIDS-derivatives files). You can also pass the names of individual derivatives pipelines (e.g., passing `'fmriprep'` would search only in a `/derivatives/fmriprep` folder). Either a string or a list of strings can be passed.

The following call returns the filenames of all derivatives files.

```{code-cell} ipython3
# Get all files in derivatives
layout2.get(scope='derivatives', return_type='file')
```

### Exporting a `BIDSLayout` to a pandas `Dataframe`
If you want a summary of all the files in your `BIDSLayout`, but don't want to have to iterate `BIDSFile` objects and extract their entities, you can get a nice bird's-eye view of your dataset using the `to_df()` method.

```{code-cell} ipython3
# Convert the layout to a pandas dataframe
df = layout.to_df()
df.head()
```

We can also include metadata in the result if we like (which may blow up our `DataFrame` if we have a large dataset). Note that in this case, most of our cells will have missing values.

```{code-cell} ipython3
layout.to_df(metadata=True).head()
```

## Retrieving BIDS variables 
BIDS variables are stored in .tsv files at the run, session, subject, or dataset level. You can retrieve these variables with `layout.get_collections()`. The resulting objects can be converted to dataframes and merged with the layout to associate the variables with corresponding scans.

In the following example, we request all subject-level variable data available anywhere in the BIDS project, and merge the results into a single `DataFrame` (by default, we'll get back a single `BIDSVariableCollection` object for each subject). 

```{code-cell} ipython3
# Get subject variables as a dataframe and merge them back in with the layout
subj_df = layout.get_collections(level='subject', merge=True).to_df()
subj_df.head()
```

## BIDSValidator

`pybids` implicitly imports a `BIDSValidator` class from the separate [`bids-validator`](https://github.com/bids-standard/bids-validator) package. You can use the `BIDSValidator` to determine whether a filepath is a valid BIDS filepath, as well as answering questions about what kind of data it represents. Note, however, that this implementation of the BIDS validator is *not* necessarily up-to-date with the JavaScript version available online. Moreover, the Python validator only tests individual files, and is currently unable to validate entire BIDS datasets. For that, you should use the [online BIDS validator](https://bids-standard.github.io/bids-validator/).

```{code-cell} ipython3
from bids import BIDSValidator

# Note that when using the bids validator, the filepath MUST be relative to the top level bids directory
validator = BIDSValidator()
validator.is_bids('/sub-02/ses-01/anat/sub-02_ses-01_T2w.nii.gz')
```

```{code-cell} ipython3
# Can decide if a filepath represents a file part of the specification
validator.is_file('/sub-02/ses-01/anat/sub-02_ses-01_T2w.json')
```

```{code-cell} ipython3
# Can check if a file is at the top level of the dataset
validator.is_top_level('/dataset_description.json')
```

```{code-cell} ipython3
# or subject (or session) level
validator.is_subject_level('/dataset_description.json')
```

```{code-cell} ipython3
validator.is_session_level('/sub-02/ses-01/sub-02_ses-01_scans.json')
```

```{code-cell} ipython3
# Can decide if a filepath represents phenotypic data
validator.is_phenotypic('/sub-02/ses-01/anat/sub-02_ses-01_T2w.nii.gz')
```
