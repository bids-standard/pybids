# Changelog

## Version 0.9.0 (May ??, 2019)
Version 0.9 replaces the native Python backend with a SQLite database managed
via SQLAlchemy. The layout module has been refactored (again), but API changes
are minimal. This release also adds many new features and closes a number of
open issues.

API CHANGES/DEPRECATIONS:
* The `extensions` argument has now been banished forever; instead, use
`extension`, which is now defined as a first-class entity. The former will
continue to work until at least the 0.11 release (closes #404).
* Relatedly, values for `extension` should no longer include a leading `.`,
though this should also continue to work for the time being.
* The `BIDSLayout` init argument `index_associated` has been removed as the
various other filtering/indexing options mean there is longer a good reason for
users to manipulate this.
* `bids.layout.MetadataIndex` no longer exists. It's unlikely that anyone will
notice this.

NEW FUNCTIONALITY:
* All file and metadata indexing and querying is now supported by a
relational (SQLite) database (see #422). While this has few API implications,
the efficiency of many operations is improved, and complex user-generated
queries can now be performed via the SQLAlchemy `.session` stored in each
`BIDSLayout`.
* Adds `.save()` method to the `BIDSLayout` that saves the current SQLite DB
to the specified location. Conversely, passing a filename as `database_file` at
init will use the specified store instead of re-indexing all files. This
eliminates the need for a pickling strategy (#435).
* Related to the above, the `BIDSLayout` init adds a `reset_database` argument
that forces reindexing even if a `database_file` is specified.
* The `BIDSLayout` has a new `index_metadata` flag that controls whether or
not the contents of JSON metadata files are indexed.
* Added `metadata` flag to `BIDSLayout.to_df()` that controls whether or not
metadata columns are included in the returned pandas `DataFrame` (#232).
* Added `get_entities()` method to `BIDSLayout` that allows retrieval of all
`Entity` instances available within a specified scope (#346).
* Adds `drop_invalid_filters` argument to `BIDSLayout.get()`, enabling users to
(optionally) ensure that invalid filters don't clobber all search results
(#402).
* `BIDSFile` instances now have a `get_associations()` method that returns
associated files (see #431).
* The `BIDSFile` class has been split into a hierarchy, with `BIDSImageFile`
and `BIDSDataFile` subclasses. The former adds a `get_image()` method (returns
a NiBabel image); the latter adds a `get_df()` method (returns a pandas DF).

BUG FIXES AND OTHER MINOR CHANGES:
* Metadata key/value pairs and file entities are now treated identically,
eliminating a source of ambiguity in search (see #398).
* Metadata no longer bleeds between raw and derivatives directories unless
explicitly specified (see #383).
* `BIDSLayout.get_collections()` no longer drops user-added columns (#273).
* Various minor fixes/improvements/changes to tests.

## Version 0.8.0 (February 15, 2019)
Version 0.8 refactors much of the layout module. It drops the grabbit
dependency, overhauls the file indexing process, and features a number of other
improvements. However, changes to the public API are very minimal, and in the
vast majority of cases, 0.8 should be a drop-in replacement for 0.7.*.

API-BREAKING CHANGES:
* Changes to (rarely-used) BIDSLayout initialization arguments:
	* `include` and `exclude` have been replaced with `ignore` and
	`force_index`. Paths passed to `ignore` will be ignored from indexing;
	paths passed to `force_index` will be forcibly indexed even if they are
	otherwise BIDS-non-compliant. `force_index` takes precedence over `ignore`.
* Most querying/selection methods add a new `scope` argument that controls
scope of querying (e.g., `'raw'`, `'derivatives'`, `'all'`, etc.). In some
cases this replaces the more limited `derivatives` argument.
* No more `domains`: with the grabbit removal (see below), the notion of a
`'domain'` has been removed. This should impact few users, but those who need
to restrict indexing or querying to specific parts of a BIDS project should be
able to use the `scope` argument more effectively.

OTHER CHANGES:
* FIX: Path indexing issues in `get_file()` (#379)
* FIX: Duplicate file returns under certain conditions (#350)
* FIX: Pass new variable args as kwargs in split() (#386) @effigies
* TEST: Update naming conventions for synthetic dataset (#385) @effigies
* REF: The grabbit package is no longer a dependency; as a result, much of the
functionality from grabbit has been ported over to pybids.
* REF: Required functionality from six and inflect is now bundled with pybids
in `bids.external`, minimizing external dependencies.
* REF: Core modules have been reorganized. Key data structures and containers
(e.g., `BIDSFile`, `Entity`, etc.) are now in a new `bids.layout.core` module.
* REF: A new `Config` class has been introduced to house the information
found in `bids.json` and other layout configuration files.
* REF: The file-indexing process has been completely refactored. A new
hierarchy of `BIDSNode` objects has been introduced. While this has no real
impact on the public API, and isn't really intended for public consumption yet,
it will in future make it easier for users to work with BIDS projects in a
tree-like way, while also laying the basis for a more sensible approach to
reading and accessing associated BIDS data (e.g., .tsv files).
* MNT: All invocations of `pd.read_table` have been replaced with `read_csv`.

## Version 0.7.1 (February 01, 2019)

This is a bug fix release in the 0.7 series. The primary API change is improved
handling of `Path` objects.

* FIX: Path validation (#342)
* FIX: Ensure consistent entities at all levels (#326)
* FIX: Edge case where a resampled column was too-long-by-one (#365)
* FIX: Use BIDS metadata for TR over nii header (#357)
* FIX: Add check for `run_info` to be a list, pass `run_info` in correct position. (#353)
* FIX: If `sampling_rate` is `'auto'`, set to first rate of `DenseRunVariables` (#351)
* FIX: Get the absolute path of the test data directory (#347)
* FIX: Update reports to be 0.7-compatible (#341)
* ENH: Rename `sr` variable to more intuitive `interval` (#366)
* ENH: Support `pathlib.Path` and other `str`-castable types (#307)
* MNT: Updates link to derivative config file in notebook (#344)
* MNT: Add bids-validator dependency (#363)
* MNT: Require pandas >= 0.23.0 (#348)
* MNT: Bump grabbit version (#338)
* CI: Ignore OSX Python 3.5 failures (#372)
* CI: Build with Python 3.7 on Travis, deploy on 3.6 (#337)

## Version 0.7.0 (January 10, 2019)

This is a major, API-breaking release. It introduces a large number of new features, bug fixes, and improvements.

API-BREAKING CHANGES:
* A number of entities (or keywords) have been renamed to align more closely with the BIDS specification documents:
	* 'type' becomes 'suffix'
	* 'modality' becomes 'datatype'
	* 'acq' is removed (use 'acquisition')
	* 'mod' becomes 'modality'
* The following directories are no longer indexed by default: derivatives/, code/, stimuli/, models/, sourcedata/. They must be explicitly included using the 'include' initialization argument.
* The grabbids module has been renamed to layout and BIDSLayout.py and BIDSvalidator.py are now layout.py and validation.py, respectively.
* The BIDS validator is now enabled by default at layout initialization (i.e., `validate=True`)
* The `exclude` initialization argument has been removed.
* `BIDSLayout.parse_entities` utility has been removed (use the more flexible `parse_file_entities`).
* Calls to `.get()` now return `BIDSFile` objects, rather than namedtuples, by default (#281). The `BIDSFile` API has been tweaked to ensure backwards incompatibility in nearly all cases.
* Naming conventions throughout the codebase have been updated to ensure consistency with the BIDS specs. This is most salient in the `analysis` module, where snake_case has been replaced with CamelCase throughout.

NEW FEATURES:
* File metadata (i.e., in JSON sidecars) is now searchable by default, and behaves just like native BIDS entities (e.g., metadata keys can be passed as arguments to `.get()` calls)
* A new BIDSFile wrapper provides easy access to `.metadata` and `.image`
* HRF convolution is now supported via bundling of nistats' hemodynamic_models module; convolution is handled via the `convolve_HRF` transformation.
* Named config paths that customize how projects are processed can be added at run-time (#313)
* Preliminary support for BIDS-Derivatives RC1 (mainly core keywords)

MINOR IMPROVEMENTS AND BUG FIXES:
* Specifying 'derivatives' in a path specification now automatically includes 'bids' (#246)
* Zenodo DOIs are now minted with new releases (#308)
* Variable loading via load_variables can now be done incrementally
* Expanded and improved path-building via `layout.build_path()`
* `get_collections` no longer breaks when `merge=True` and the list is empty (#202)
* Layout initialization no longer fails when `validate=True` (#222)
* The auto_contrasts field in the modeling tools now complies with the BIDS-Model spec (#234)
* Printing a `BIDSFile` now provides more useful information, including path (#298)
* Resample design matrix to 1/TR by default (#309)
* Fix the Sum transformation
* Ensure that resampling works properly when a sampling rate is passed to `get_design_matrix` (#297)
* Propagate derivative entities into top-level dynamic getters (#306)
* Deprecated `get_header` call in nibabel removed (#300)
* Fix bug in entity indexing for `BIDSVariableCollection` (#319)
* Exclude modules with heavy dependencies from root namespace for performance reasons (#321)
* Fix bug that caused in-place updating of input selectors in `Analysis` objects (#323)
* Add a DropNA transformation (#325)
* Add a `get_tr()` method to `BIDSLayout` (#327)
* Add entity hints when calling `get()` with a `target` argument (#328)
* Improved test coverage

## Version 0.6.5 (August 21, 2018)

* FIX: Do not drop rows of NaNs (#217) @adelavega
* FIX: Declare run as having integer type (#236) @effigies
* ENH: MEG support (#229) @jasmainak
* REF: rename grabbids to layout, closes #228 (#230) @ltirrell
* DOC: add .get_collection examples to tutorial (#219) @Shotgunosine
* DOC: Fix link in README to point to documentation (#223) @KirstieJane
* DOC: Add binder link for tutorial (#225) @KirstieJane
* MAINT: Restore "analysis" installation extra (#218) @yarikoptic
* MAINT: Do not import tests in \_\_init\_\_.py (#226) @tyarkoni

## Version 0.5.1 (March 9, 2018)
Hotfix release:

* Includes data files omitted from 0.5.0 release.
* Improves testing of installation.

## Version 0.5.0 (March 6, 2018)
This is a major release that introduces the following features:
* A new `bids.variables` module that adds the following submodules:
	* `bids.variables.entities.py`: Classes for representing BIDS hierarchies as a graph-like structure.
	* `bids.variables.variables.py`: Classes and functions for representing and manipulating non-imaging data read from BIDS projects (e.g., fMRI events, densely-sampled physiological measures, etc.).
	* `bids.variables.io.py`: Tools for loading variable data from BIDS projects.
	* `bids.variables.kollekshuns`: Containers that facilitate aggregation and manipulation of `Variable` classes.
* Extensions to the `BIDSLayout` class that make it easy to retrieve data/variables from the project (i.e., `Layout.get_collections`)
* A new `auto_model` utility that generates simple BIDS-Model specifications from BIDS projects (thanks to @Shotgunosine)
* A new `reports` module that generates methods sections from metadata in BIDS projects (thanks to @tsalo)
* Experimental support for copying/writing out files in a BIDS-compliant way
* Expand `bids.json` config file to include missing entity definitions
* Ability to parse files without updating the Layout index
* Updated grabbids module to reflect grabbit changes that now allow many-to-many mapping of configurations to folders
* Too many other minor improvements and bug fixes to list (when you're very lazy, even a small amount of work is too much)

## Version 0.4.2 (November 16, 2017)
We did some minor stuff, but we were drunk again and couldn't read our handwriting on the napkin the next morning.

## Version 0.4.1 (November 3, 2017)
We did some minor stuff, and we didn't think it was important enough to document.

## Version 0.4.0 (November 1, 2017)
We did some stuff, but other stuff was happening in the news, and we were too distracted to write things down.

## Version 0.3.0 (August 11, 2017)
We did some stuff, but we were drunk and forgot to write it down.

## Version 0.2.1 (June 8, 2017)
History as we know it begins here.
