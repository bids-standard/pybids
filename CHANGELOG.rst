Changelog
=========

Version 0.15.0 (March 28, 2022)
-------------------------------

New feature release in the 0.15.x series.

Downstream tools should be aware of a potentially breaking, albeit
long-demanded, change introduced in #819. Run indices are now stored
so that the integers that come out retain any zero-padding that was
found during parsing.

This release also introduces the ``bids.ext`` namespace package that
allows independent packages to install modules in this namespace.
This is an infrastructural change that will allow some components to
be separately managed and follow a different development pace.

* FIX: Allow grouping by run and session when entities are undefined (#822)
* FIX: Clarify exception message (#806)
* FIX: Catch UnicodeDecodeErrors along with JSONDecodeErrors for better reporting (#796)
* FIX: Accept paths/strings for layout configuration files (#799)
* FIX: Small typo: repeated word in docstring (#793)
* ENH: Retain zero-padding in run entities while preserving integer queries and comparisons (#819)
* ENH: Add bids.ext namespace package for subpackages (#820)
* ENH: Handle wildcards in model X (#810)
* ENH: Implement automatic detection of derivative data (#805)
* ENH: Add new ``Query`` for optional entities (#809)
* ENH: Add __main__ to allow ``python -m bids`` to run CLI (#794)
* REF: Improve modularization of bids.reports (#617)
* DOC: Link from sphinx documentation to notebook tutorials. (#797)
* MNT: Test on Python 3.10, various CI updates (#824)
* MNT: Avoid jinja2 v3 until nbconvert handles breakages (#823)

Version 0.14.1 (March 29, 2022)
-------------------------------
Bug-fix release in the 0.14.x series.

* RF/FIX: Decompose filter construction for special queries and lists (#826)

Includes the following back-ports from 0.15.0:

* FIX: Clarify exception message (#806)
* FIX: Catch UnicodeDecodeErrors along with JSONDecodeErrors for better reporting (#796)
* FIX: Accept paths/strings for layout configuration files (#799)
* ENH: Add __main__ to allow ``python -m bids`` to run CLI (#794)

Version 0.14.0 (November 09, 2021)
----------------------------------

New feature release in the 0.14.x series.

This release includes a significant refactor of BIDS Statistical Models,
replacing the ``bids.analysis`` module with ``bids.modeling``.

Changes to ``bids.layout`` are minimal, and we do not anticipate API breakage.

* FIX: LGTM.com warning: Implicit string concatenation in a list (#785)
* FIX: Take the intersection of variables and Model.X,
  ignoring missing variables (usually contrasts) (#764)
* FIX: Associate "is_metadata" with Tag, not Entity; and only return
  non-metadata entries for core Entities in ``get(return_type='id')`` (#749)
* FIX: Only include regressors if they are TSV (#752)
* FIX: ensure force_dense=True runs to_dense only on sparse variables (#745)
* FIX: get unique, with conflicting meta-data (#748)
* FIX: Clean up some deprecation and syntax warnings (#738)
* ENH: Add ``pybids upgrade`` command (#654)
* ENH: Add Lag transformation (#759)
* ENH: Use indirect transformations structure (#737)
* ENH: Add visualization for statsmodel graph (#742)
* ENH: Permit explicit intercept (1) in Contrasts and DummyContrasts (#743)
* ENH: Add meta-analysis model type (#731)
* ENH: Contrast type is now test (#733)
* REF: Use pathlib.Path internally when possible (#746)
* REF: Remove group_by from edges and add filter (#734)
* REF: Improved/refactored StatsModels module (#722)
* MNT: Make sure codespell skips .git when run locally (#787)
* MNT: LGTM.com recommendations (#786)
* MNT: Better codespell configuration (#782)
* MNT: Constrain formulaic version to 0.2.x . (#784)
* MNT: Update versioneer: 0.18 → 0.20 (#778)
* MNT: Add "codespell" tool to CI checks to catch typos sooner (#776)
* MNT: Disable bids-nodot mode (#769)
* MNT: Send codecov reports again (#766)
* MNT: Set minimum version to Python 3.6 (#739)

Version 0.13.2 (August 20, 2021)
--------------------------------

Bug-fix release in the 0.13 series.

* FIX/TEST: gunzip regressors.tsv.gz, allow timeseries.tsv as well (#767)
* FIX: run is required (#762)
* MNT: Patch 0.13.x maint branch (#763)

Version 0.13.1 (May 21, 2021)
-----------------------------

Bug-fix release in the 0.13 series.

* ENH: Improve `get` performance (#723)
* STY: Fix typos identified by codespell (#720)
* TEST: dataset-level model spec retrieval (#693)

Version 0.13 (April 14, 2021)
-----------------------------

New feature release in the 0.13 series.

* FIX: Resample to n_vols for sampling_rate == 'TR' (#713)
* FIX: Lazily load metadata, skip when missing data files are missing sidecars (#711)
* FIX: Ensure indicator matrix is boolean when indexing in Split transformation (#710)
* FIX: Correctly pair metadata file when creating association records (#699)
* FIX: Resolve side-effects of new testfile in #682 (#695)
* FIX: BIDSLayout -- TypeError: unhashable type: 'dict' (#682)
* ENH: Add res/den entities to derivatives.json (#709)
* ENH: Update datatypes (#708)
* ENH: add more informative validation error message for dataset_description.json (#705)
* ENH: Add flip, inv, mt, and part entities (#688)
* CI: Run packaging tests on main repository only (#696)
* CI: Migrate to GH actions (#691)

Version 0.12.4 (November 10, 2020)
----------------------------------

Bug-fix release in the 0.12.x series.

* FIX: Do not error when popping missing ``scan_length`` (#679)

Version 0.12.3 (October 23, 2020)
---------------------------------

* FIX: Require aligned input for logical operations (#649)
* ENH: Incremental variable loading in Step setup (#672)

Version 0.12.2 (October 09, 2020)
---------------------------------

Bug-fix release in 0.12.x series.

* FIX: Support nibabel < 3 when calculating time series length (#669)
* FIX: Sanitize single derivative Path passed to BIDSLayout (#665)
* FIX: Force UTF-8 encoding in _index_metadata (#663)
* FIX: Explicitly convert to HTML when testing tutorial (nbconvert 6.0 breakage) (#664)

Version 0.12.1 (September 09, 2020)
-----------------------------------

* FIX: drop obsolete test (#652)
* FIX: Convolve zero-duration (impulse) events when variable contains multiple events (#645)
* ENH: Add CLI to PyBIDS (#650)
* ENH: Add relpath attribute to BIDSFile and associated refactoring (#647)
* DOC: Add example for using parse_file_entities from bids.layout (#639)
* MNT: Remove deprecated database_file argument to BIDSLayout (#644)

Version 0.12.0 (August 04, 2020)
--------------------------------
New feature release for the 0.12.x series.

This release includes significant speedups for ``BIDSLayout`` creation and improves
handling of GIFTI and CIFTI-2 derivative files.

* FIX: Remove need to eager load associations (#642)
* ENH: Fetch number of time points from NIfTI, GIFTI or CIFTI-2 (#637)
* ENH: Catch any NIfTI/GIFTI (incl. CIFTI-2) files as BIDSImageFiles (#638)

Version 0.11.1 (July 02, 2020)
------------------------------
Bug-fix release in the 0.11.x series

With thanks to Mathias Goncalves for finding and fixing the issue.

* FIX: Preserve ``get`` logic when using custom config (#636)

Version 0.11.0 (June 29, 2020)
------------------------------
New feature release in the 0.11.x series.

One significant change in this PR is to add the configuration option ``extension_initial_dot``.
Setting to ``True`` will adopt the future default behavior of returning ``extension`` entities with
an initial dot (``.``).

Other notable changes include a significant refactoring of the Analysis module, and a number of
small improvements to error reporting that should add up to simpler debugging for users and
developers.

* FIX: Post-fix And and Or transformations with underscore (#628)
* FIX: made _validate_file work on Windows (#627)
* FIX: Scale transformation fails if passed constant input. (#614)
* FIX: Certain queries involving multiple layouts are very slow (#616)
* FIX: Bug in get() when passing enums as extensions (#612)
* FIX: Bug in BIDSDataFile.get_df() (#611)
* FIX: Make entity assertions Python 3.5-safe (#606)
* FIX: BIDSLayout.build_path to respect absolute_paths. (#580)
* ENH: Adds OS-level file copying instead of reading/writing via Python (#613)
* ENH: Add explicit warning message when users pass in filters as a dictionary keyword (#623)
* ENH: Introduce PyBIDS exceptions (#615)
* ENH: Add example bids and derivatives dataset_description.json strings to error messages (#620)
* ENH: Improved handling of invalid filters (#610)
* ENH: Add method to generate report from list of files (#607)
* ENH: Pass kwargs from BIDSImageFile.get_image() to nibabel.load (#601)
* ENH: Model spec module and associated utilities (#548)
* ENH: Add BIDSMetadata dictionary to report file with missing metadata (#593)
* RF: Add ``extension_initial_dot`` config option to transition to extension entity with initial dot (#629)
* MNT: Automatically deploy docs (#598)
* CI: Drop --pre check for Python 3.5 (#621)
* CI: Test on Python 3.8 (#594)

Version 0.10.2 (February 26, 2020)
----------------------------------
Bug fix release in the 0.10.x series.

* FIX: Add Replace as exception to recursive JSON conversion (#589)

Version 0.10.1 (February 10, 2020)
----------------------------------
Bug fix release in the 0.10.x series.

This release just makes available some of the latest minor fixes and improvements.

* FIX: Replace ``os.path.sep`` with ``fwdslash`` because bids validator hardcodes fwd (#582)
* FIX: Refactor of ``build_path`` and inner machinery (#574)
* FIX: Domain entity, and slow ``__repr__`` (#569)
* FIX: "strict" helptext in ``BIDSLayout.write_contents_to_file`` (#566)
* FIX: typos in helpstrings and comments (#564)
* FIX: Correct term "caret" to "angle bracket" in helpstrings (#565)
* ENH: Extend ``build_path`` to generate lists of files (#576)
* ENH: Add one parametric test of ``BIDSLayout.build_path`` (#577)
* ENH: Enable partial metadata indexing (#560)
* ENH: Upscale to collection sampling rate prior to resampling (#568)
* ENH: Calculate default sampling rate for ``SparseRunVariable.to_dense`` (#571)
* MNT: Add ``.vscode`` (for Visual Studio Code) to ``.gitignore`` (#562)
* MNT: Ignore pip-wheel-metadata (#581)
* DOC: Remove Python 2 support statement, now that v0.10.0 has dropped it (#561)

Version 0.10.0 (December 03, 2019)
----------------------------------
New feature release in the 0.10.x series.

This release removes Python 2 support.

* ENH: Helpful error for db argument mismatch, and add classmethod load_from_db (#547)
* ENH: Add Resample transformation (#373)
* ENH: Save BIDSLayout + Derivatives to folder (with init arguments) (#540)
* ENH: Adds support for NONE and ANY query Enums (#542)
* ENH: Transformation-related improvements (#541)
* ENH: FEMA contrasts (#520)
* STY: Minor PEP8 Fixes (#545)
* MNT: Various (#543)
* MNT: Remove Python 2.7 support (#524)
* CI: Configure Circle Artifact Redirector (#544)

Version 0.9.5 (November 6, 2019)
--------------------------------
Bug fix release in the 0.9.x series.

Final planned release with Python 2 support.

* FIX: Filter before downsampling (#529)
* FIX: Copy input dict in ``replace_entities``\ ; Typos in ``default_path_patterns`` (#517)
* FIX: Use string dtype for all entities when using regex search (#511)
* FIX: Update Sphinx docs for 2.2 (#507)
* ENH: Enable automatic derivative database caching (#523)
* ENH: Raise ValueError if BIDSLayout.build_path fails to match any pattern (#508)
* RF: Subclass analysis Node from object (#528)
* DOC: Unify docstring format (#499)
* DOC: Auto-generate stubs (#513)
* STY: .sql is a misleading extension to sqlite files (#531)
* STY: General cleanups (#526, #527)

Version 0.9.4 (September 20, 2019)
----------------------------------
Bug fix release in the 0.9.x series.

* FIX: Ignore ``default_ignore`` paths by default (#495)
* FIX: Filter and sort on scalar metadata in ``get_nodes()`` (#488)
* FIX: Automatically sanitize dtype of ``get()`` arguments (#492)
* FIX: Check that ``default_path_patterns`` is not ``None`` before using in ``build_path`` (#485)
* FIX: Add CBV and phase modalities to func datatype path pattern (#479)
* FIX: Drop bold suffix constraint from echo entity (#477)
* ENH: Enforce dtypes on spec-defined columns when reading in DFs (#494)
* ENH: Validate built paths (#496)
* ENH: Update contrast enhancing agent query name (ceagent) (#497)
* DOC: Add citation information to README (#493)
* DOC: Improve wording in Python notebook example comment (#484)
* DOC: Finish sentence in Python Notebook example documentation (#483)
* DOC: Add JOSS Badge (#472)
* STY: Apply syntax highlight to Python notebook example doc (#482)
* MAINT: Move setup configuration to setup.cfg (#448)
* MAINT: Additional Zenodo metadata (#470)
* MAINT/CI: Use ``extras_require`` to declare dependencies for tests  (#471)

Version 0.9.3 (August 7, 2019)
------------------------------
This version includes a number of minor fixes and improvements, one of which
breaks the existing API (by renaming two entities; see #464). Changes
include:

* FIX: Avoid DB collisions for redundant entities (#468)
* FIX: Minor changes to entity names in core spec (#464)
* FIX: Make bids.reports work properly with .nii images (#463)
* CI: Execute notebook in Travis (#461)
* ENH: More sensible **repr** for Tag model (#467)

Version 0.9.2 (July 12, 2019)
-----------------------------
This version includes a number of minor fixes and improvements.
EEG files are better handled, and ``BIDSLayout`` and ``BIDSFile`` play more
nicely with ``Path``\ -like objects.

With thanks to new contributor Cecile Madjar.

* FIX: Instantiate ``ignore``\ /\ ``force_index`` after root validation (#457)
* FIX: Restore ``<entity>=None`` query returning files lacking the entity (#458)
* ENH: Add ``BIDSJSONFile`` (#444)
* ENH: Add ``BIDSFile.__fspath__`` to work with pathlib (#449)
* ENH: Add ``eeg`` datatype to layout config (#455)
* RF: Remove unused kwargs to BIDSFile (#443)
* DOC: Improve docstring consistency, style (#443)
* DOC: Address final JOSS review (#453)
* STY: PEP8 Fixes (#456)
* MAINT: Set name explicitly in setup.py (#450)

Version 0.9.1 (May 24, 2019)
----------------------------
Hotfix release:

* Fixed package deployment issues (#446)
* Updated author list (#447)

Thanks to new contributors Erin Dickie, Chadwick Boulay and Johannes Wennberg.

Version 0.9.0 (May 21, 2019)
----------------------------
Version 0.9 replaces the native Python backend with a SQLite database managed
via SQLAlchemy. The layout module has been refactored (again), but API changes
are minimal. This release also adds many new features and closes a number of
open issues.

API CHANGES/DEPRECATIONS:

* The ``extensions`` argument has now been banished forever; instead, use
  ``extension``\ , which is now defined as a first-class entity. The former will
  continue to work until at least the 0.11 release (closes #404).
* Relatedly, values for ``extension`` should no longer include a leading ``.``\ ,
  though this should also continue to work for the time being.
* The ``BIDSLayout`` init argument ``index_associated`` has been removed as the
  various other filtering/indexing options mean there is longer a good reason for
  users to manipulate this.
* ``bids.layout.MetadataIndex`` no longer exists. It's unlikely that anyone will
  notice this.
* ``BIDSLayout.get_metadata()`` no longer takes additional entities as optional
  keyword arguments (they weren't necessary for anything).
* Direct access to most ``BIDSFile`` properties is discouraged, and in one case
  is broken in 0.9 (for ``.metadata``\ , which was unavoidable, because it's reserved
  by SQLAlchemy). Instead, users should use getters (\ ``get_metadata``\ , ``get_image``\ ,
  ``get_df``\ , etc.) whenever possible.

NEW FUNCTIONALITY:

* All file and metadata indexing and querying is now supported by a
  relational (SQLite) database (see #422). While this has few API implications,
  the efficiency of many operations is improved, and complex user-generated
  queries can now be performed via the SQLAlchemy ``.session`` stored in each
  ``BIDSLayout``.
* Adds ``.save()`` method to the ``BIDSLayout`` that saves the current SQLite DB
  to the specified location. Conversely, passing a filename as ``database_file`` at
  init will use the specified store instead of re-indexing all files. This
  eliminates the need for a pickling strategy (#435).
* Related to the above, the ``BIDSLayout`` init adds a ``reset_database`` argument
  that forces reindexing even if a ``database_file`` is specified.
* The ``BIDSLayout`` has a new ``index_metadata`` flag that controls whether or
  not the contents of JSON metadata files are indexed.
* Added ``metadata`` flag to ``BIDSLayout.to_df()`` that controls whether or not
  metadata columns are included in the returned pandas ``DataFrame`` (#232).
* Added ``get_entities()`` method to ``BIDSLayout`` that allows retrieval of all
  ``Entity`` instances available within a specified scope (#346).
* Adds ``drop_invalid_filters`` argument to ``BIDSLayout.get()``\ , enabling users to
  (optionally) ensure that invalid filters don't clobber all search results
  (#402).
* ``BIDSFile`` instances now have a ``get_associations()`` method that returns
  associated files (see #431).
* The ``BIDSFile`` class has been split into a hierarchy, with ``BIDSImageFile``
  and ``BIDSDataFile`` subclasses. The former adds a ``get_image()`` method (returns
  a NiBabel image); the latter adds a ``get_df()`` method (returns a pandas DF).
  All ``BIDSFile`` instances now also have a ``get_entities()`` method.

BUG FIXES AND OTHER MINOR CHANGES:

* Metadata key/value pairs and file entities are now treated identically,
  eliminating a source of ambiguity in search (see #398).
* Metadata no longer bleeds between raw and derivatives directories unless
  explicitly specified (see #383).
* ``BIDSLayout.get_collections()`` no longer drops user-added columns (#273).
* Various minor fixes/improvements/changes to tests.
* The tutorial Jupyter notebook has been fixed and updated to reflect the
  latest changes.

Version 0.8.0 (February 15, 2019)
---------------------------------
Version 0.8 refactors much of the layout module. It drops the grabbit
dependency, overhauls the file indexing process, and features a number of other
improvements. However, changes to the public API are very minimal, and in the
vast majority of cases, 0.8 should be a drop-in replacement for 0.7.*.

API-BREAKING CHANGES:

* Changes to (rarely-used) BIDSLayout initialization arguments:
  * ``include`` and ``exclude`` have been replaced with ``ignore`` and
    ``force_index``. Paths passed to ``ignore`` will be ignored from indexing;
    paths passed to ``force_index`` will be forcibly indexed even if they are
    otherwise BIDS-non-compliant. ``force_index`` takes precedence over ``ignore``.
* Most querying/selection methods add a new ``scope`` argument that controls
  scope of querying (e.g., ``'raw'``\ , ``'derivatives'``\ , ``'all'``\ , etc.). In some
  cases this replaces the more limited ``derivatives`` argument.
* No more ``domains``\ : with the grabbit removal (see below), the notion of a
  ``'domain'`` has been removed. This should impact few users, but those who need
  to restrict indexing or querying to specific parts of a BIDS project should be
  able to use the ``scope`` argument more effectively.

OTHER CHANGES:

* FIX: Path indexing issues in ``get_file()`` (#379)
* FIX: Duplicate file returns under certain conditions (#350)
* FIX: Pass new variable args as kwargs in split() (#386) @effigies
* TEST: Update naming conventions for synthetic dataset (#385) @effigies
* REF: The grabbit package is no longer a dependency; as a result, much of the
  functionality from grabbit has been ported over to pybids.
* REF: Required functionality from six and inflect is now bundled with pybids
  in ``bids.external``\ , minimizing external dependencies.
* REF: Core modules have been reorganized. Key data structures and containers
  (e.g., ``BIDSFile``\ , ``Entity``\ , etc.) are now in a new ``bids.layout.core`` module.
* REF: A new ``Config`` class has been introduced to house the information
  found in ``bids.json`` and other layout configuration files.
* REF: The file-indexing process has been completely refactored. A new
  hierarchy of ``BIDSNode`` objects has been introduced. While this has no real
  impact on the public API, and isn't really intended for public consumption yet,
  it will in future make it easier for users to work with BIDS projects in a
  tree-like way, while also laying the basis for a more sensible approach to
  reading and accessing associated BIDS data (e.g., .tsv files).
* MNT: All invocations of ``pd.read_table`` have been replaced with ``read_csv``.

Version 0.7.1 (February 01, 2019)
---------------------------------
This is a bug fix release in the 0.7 series. The primary API change is improved
handling of ``Path`` objects.

* FIX: Path validation (#342)
* FIX: Ensure consistent entities at all levels (#326)
* FIX: Edge case where a resampled column was too-long-by-one (#365)
* FIX: Use BIDS metadata for TR over nii header (#357)
* FIX: Add check for ``run_info`` to be a list, pass ``run_info`` in correct position. (#353)
* FIX: If ``sampling_rate`` is ``'auto'``\ , set to first rate of ``DenseRunVariables`` (#351)
* FIX: Get the absolute path of the test data directory (#347)
* FIX: Update reports to be 0.7-compatible (#341)
* ENH: Rename ``sr`` variable to more intuitive ``interval`` (#366)
* ENH: Support ``pathlib.Path`` and other ``str``\ -castable types (#307)
* MNT: Updates link to derivative config file in notebook (#344)
* MNT: Add bids-validator dependency (#363)
* MNT: Require pandas >= 0.23.0 (#348)
* MNT: Bump grabbit version (#338)
* CI: Ignore OSX Python 3.5 failures (#372)
* CI: Build with Python 3.7 on Travis, deploy on 3.6 (#337)

Version 0.7.0 (January 10, 2019)
--------------------------------
This is a major, API-breaking release. It introduces a large number of new features, bug fixes, and improvements.

API-BREAKING CHANGES:

* A number of entities (or keywords) have been renamed to align more closely with the BIDS specification documents:
  * 'type' becomes 'suffix'
  * 'modality' becomes 'datatype'
  * 'acq' is removed (use 'acquisition')
  * 'mod' becomes 'modality'
* The following directories are no longer indexed by default: derivatives/, code/, stimuli/, models/, sourcedata/. They must be explicitly included using the 'include' initialization argument.
* The grabbids module has been renamed to layout and BIDSLayout.py and BIDSvalidator.py are now layout.py and validation.py, respectively.
* The BIDS validator is now enabled by default at layout initialization (i.e., ``validate=True``\ )
* The ``exclude`` initialization argument has been removed.
* ``BIDSLayout.parse_entities`` utility has been removed (use the more flexible ``parse_file_entities``\ ).
* Calls to ``.get()`` now return ``BIDSFile`` objects, rather than namedtuples, by default (#281). The ``BIDSFile`` API has been tweaked to ensure backwards incompatibility in nearly all cases.
* Naming conventions throughout the codebase have been updated to ensure consistency with the BIDS specs. This is most salient in the ``analysis`` module, where snake_case has been replaced with CamelCase throughout.

NEW FEATURES:

* File metadata (i.e., in JSON sidecars) is now searchable by default, and behaves just like native BIDS entities (e.g., metadata keys can be passed as arguments to ``.get()`` calls)
* A new BIDSFile wrapper provides easy access to ``.metadata`` and ``.image``
* HRF convolution is now supported via bundling of nistats' hemodynamic_models module; convolution is handled via the ``convolve_HRF`` transformation.
* Named config paths that customize how projects are processed can be added at run-time (#313)
* Preliminary support for BIDS-Derivatives RC1 (mainly core keywords)

MINOR IMPROVEMENTS AND BUG FIXES:

* Specifying 'derivatives' in a path specification now automatically includes 'bids' (#246)
* Zenodo DOIs are now minted with new releases (#308)
* Variable loading via load_variables can now be done incrementally
* Expanded and improved path-building via ``layout.build_path()``
* ``get_collections`` no longer breaks when ``merge=True`` and the list is empty (#202)
* Layout initialization no longer fails when ``validate=True`` (#222)
* The auto_contrasts field in the modeling tools now complies with the BIDS-Model spec (#234)
* Printing a ``BIDSFile`` now provides more useful information, including path (#298)
* Resample design matrix to 1/TR by default (#309)
* Fix the Sum transformation
* Ensure that resampling works properly when a sampling rate is passed to ``get_design_matrix`` (#297)
* Propagate derivative entities into top-level dynamic getters (#306)
* Deprecated ``get_header`` call in nibabel removed (#300)
* Fix bug in entity indexing for ``BIDSVariableCollection`` (#319)
* Exclude modules with heavy dependencies from root namespace for performance reasons (#321)
* Fix bug that caused in-place updating of input selectors in ``Analysis`` objects (#323)
* Add a DropNA transformation (#325)
* Add a ``get_tr()`` method to ``BIDSLayout`` (#327)
* Add entity hints when calling ``get()`` with a ``target`` argument (#328)
* Improved test coverage

Version 0.6.5 (August 21, 2018)
-------------------------------

* FIX: Do not drop rows of NaNs (#217) @adelavega
* FIX: Declare run as having integer type (#236) @effigies
* ENH: MEG support (#229) @jasmainak
* REF: rename grabbids to layout, closes #228 (#230) @ltirrell
* DOC: add .get_collection examples to tutorial (#219) @Shotgunosine
* DOC: Fix link in README to point to documentation (#223) @KirstieJane
* DOC: Add binder link for tutorial (#225) @KirstieJane
* MAINT: Restore "analysis" installation extra (#218) @yarikoptic
* MAINT: Do not import tests in __init__.py (#226) @tyarkoni

Version 0.5.1 (March 9, 2018)
-----------------------------
Hotfix release:

* Includes data files omitted from 0.5.0 release.
* Improves testing of installation.

Version 0.5.0 (March 6, 2018)
-----------------------------
This is a major release that introduces the following features:

* A new ``bids.variables`` module that adds the following submodules:
  * ``bids.variables.entities.py``\ : Classes for representing BIDS hierarchies as a graph-like structure.
  * ``bids.variables.variables.py``\ : Classes and functions for representing and manipulating non-imaging data read from BIDS projects (e.g., fMRI events, densely-sampled physiological measures, etc.).
  * ``bids.variables.io.py``\ : Tools for loading variable data from BIDS projects.
  * ``bids.variables.collections``\ : Containers that facilitate aggregation and manipulation of ``Variable`` classes.
* Extensions to the ``BIDSLayout`` class that make it easy to retrieve data/variables from the project (i.e., ``Layout.get_collections``\ )
* A new ``auto_model`` utility that generates simple BIDS-Model specifications from BIDS projects (thanks to @Shotgunosine)
* A new ``reports`` module that generates methods sections from metadata in BIDS projects (thanks to @tsalo)
* Experimental support for copying/writing out files in a BIDS-compliant way
* Expand ``bids.json`` config file to include missing entity definitions
* Ability to parse files without updating the Layout index
* Updated grabbids module to reflect grabbit changes that now allow many-to-many mapping of configurations to folders
* Too many other minor improvements and bug fixes to list (when you're very lazy, even a small amount of work is too much)

Version 0.4.2 (November 16, 2017)
---------------------------------
We did some minor stuff, but we were drunk again and couldn't read our handwriting on the napkin the next morning.

Version 0.4.1 (November 3, 2017)
--------------------------------
We did some minor stuff, and we didn't think it was important enough to document.

Version 0.4.0 (November 1, 2017)
--------------------------------
We did some stuff, but other stuff was happening in the news, and we were too distracted to write things down.

Version 0.3.0 (August 11, 2017)
-------------------------------
We did some stuff, but we were drunk and forgot to write it down.

Version 0.2.1 (June 8, 2017)
----------------------------
History as we know it begins here.
