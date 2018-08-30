# Changelog

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
