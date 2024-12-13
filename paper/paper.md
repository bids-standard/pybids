---
title: 'PyBIDS: Python tools for BIDS datasets'
tags:
  - Python
  - BIDS
  - neuroimaging
authors:
  - affiliation: 1
    name: Tal Yarkoni
    orcid: 0000-0002-6558-5113
  - affiliation: 2
    name: Christopher J Markiewicz
    orcid: 0000-0002-6533-164X
  - affiliation: 1
    name: Alejandro de la Vega
    orcid: 0000-0001-9062-3778
  - affiliation: 2
    name: Krzysztof J Gorgolewski
    orcid: 0000-0003-3321-7583
  - affiliation: 3
    name: Taylor Salo
    orcid: 0000-0001-9813-3167
  - affiliation: 4
    name: Yaroslav O Halchenko
    orcid: 0000-0003-3456-2493
  - affiliation: 1
    name: Quinten McNamara
  - affiliation: 5
    name: Krista DeStasio
    orcid: 0000-0002-3959-9060
  - affiliation: 6
    name: Jean-Baptiste Poline
    orcid: 0000-0002-9794-749X
  - affiliation: 7
    name: Dmitry Petrov
  - affiliation: 8
    name: Valérie Hayot-Sasson
    orcid: 0000-0002-4830-4535
  - affiliation: 9
    name: Dylan M Nielson
    orcid: 0000-0003-4613-6643
  - affiliation: 10
    name: Johan Carlin
    orcid: 0000-0003-0933-1239
  - affiliation: 11
    name: Gregory Kiar
    orcid: 0000-0001-8915-496X
  - affiliation: 12
    name: Kirstie Whitaker
    orcid: 0000-0001-8498-4059
  - affiliation: 11
    name: Elizabeth DuPre
    orcid: 0000-0003-1358-196X
  - affiliation: 13
    name: Adina Wagner
    orcid: 0000-0003-2917-3450
  - affiliation: 14
    name: Lee S Tirrell
    orcid: 0000-0001-9393-8361
  - affiliation: 15
    name: Mainak Jas
    orcid: 0000-0002-3199-9027
  - affiliation: 13
    name: Michael Hanke
    orcid: 0000-0001-6398-6370
  - affiliation: 2
    name: Russell A Poldrack
    orcid: 0000-0001-6755-0259
  - affiliation: 2
    name: Oscar Esteban
    orcid: 0000-0001-8435-6191
  - affiliation: 16
    name: Stefan Appelhoff
    orcid: 0000-0001-8002-0877
  - affiliation: 17
    name: Chris Holdgraf
    orcid: 0000-0002-2391-0678
  - affiliation: 18
    name: Isla Staden
    orcid: 0000-0002-0795-1154
  - affiliation: 19
    name: Bertrand Thirion
    orcid: 0000-0001-5018-7895
  - affiliation: 20
    name: Dave F Kleinschmidt
    orcid: 0000-0002-7442-2762
  - affiliation: 9
    name: John A Lee
    orcid: 0000-0001-5884-4247
  - affiliation: 17
    name: Matteo Visconti di Oleggio Castello
    orcid: 0000-0001-7931-5272
  - affiliation: 21
    name: Michael P Notter
    orcid: 0000-0002-5866-047X
  - affiliation: 2
    name: Ross Blair
    orcid: 0000-0003-3007-1056
affiliations:
- index: 1
  name: University of Texas at Austin
- index: 2
  name: Stanford University
- index: 3
  name: Florida International University
- index: 4
  name: Dartmouth College
- index: 5
  name: University of Oregon
- index: 6
  name: McGill University
- index: 7
  name: University of Southern California
- index: 8
  name: Concordia University
- index: 9
  name: National Institute of Mental Health
- index: 10
  name: MRC Cognition and Brain Sciences Unit
- index: 11
  name: Montreal Neurological Institute and Hospital
- index: 12
  name: Alan Turing Institute
- index: 13
  name: Otto-von-Guericke University Magdeburg
- index: 14
  name: CorticoMetrics LLC
- index: 15
  name: Télécom ParisTech, France
- index: 16
  name: Max Planck Institute for Human Development, Berlin, Germany
- index: 17
  name: University of California at Berkeley
- index: 18
  name: Queen Mary University London
- index: 19
  name: INRIA
- index: 20
  name: Rutgers University
- index: 21
  name: University of Lausanne
date: 19 February 2019
bibliography: paper.bib
---

# Summary

Brain imaging researchers regularly work with large, heterogeneous,
high-dimensional datasets. Historically, researchers have dealt with this
complexity idiosyncratically, with every lab or individual implementing their
own preprocessing and analysis procedures. The resulting lack of field-wide
standards has severely limited reproducibility and data sharing and reuse.

To address this problem, we and others recently introduced the Brain Imaging
Data Standard (``BIDS``; [@Gorgolewski2016-nw]), a specification meant to
standardize the process of representing brain imaging data. BIDS is
deliberately designed with adoption in mind; it adheres to a user-focused
philosophy that prioritizes common use cases and discourages complexity. By
successfully encouraging a large and ever-growing subset of the community to
adopt a common standard for naming and organizing files, BIDS has made it much
easier for researchers to share, reuse, and process their data
[@Gorgolewski2017-cz].

The ability to efficiently develop high-quality spec-compliant applications
itself depends to a large extent on the availability of good tooling.
Because many operations recur widely across diverse contexts—for example,
almost every tool designed to work with BIDS datasets involves regular
file-filtering operations—there is a strong incentive to develop utility
libraries that provide common functionality via a standardized, simple API.

``PyBIDS`` is a Python package that makes it easier to work with BIDS
datasets. In principle, its scope includes virtually any functionality that is
likely to be of general use when working with BIDS datasets (*i.e.*, that is not
specific to one narrow context). At present, its core and most widely used
module supports simple and flexible querying and manipulation of BIDS datasets.
PyBIDS makes it easy for researchers and developers working in Python to search
for BIDS files by keywords and/or metadata; to consolidate and retrieve
file-associated metadata spread out across multiple levels of a BIDS hierarchy;
to construct BIDS-valid path names for new files; and to validate projects
against the BIDS specification, among other applications.

In addition to this core functionality, PyBIDS also contains an ever-growing
set of modules that support additional capabilities meant to keep up with the
evolution and expansion of the BIDS specification itself. Currently, PyBIDS
includes tools for (1) reading and manipulating data contained in various
BIDS-defined files (*e.g.*, physiological recordings, event files, or
participant-level variables); (2) constructing design matrices and contrasts
that support the new ``BIDS-StatsModel`` specification (for machine-readable
representation of fMRI statistical models); and (3) automated generation of
partial Methods sections for inclusion in publications.

PyBIDS can be easily installed on all platforms via pip (``pip install
pybids``), though currently it is not officially supported on Windows. The
package has few dependencies outside of standard Python numerical and image
analysis libraries (*i.e.*, numpy, scipy, pandas, and NiBabel). The core API
is deliberately kept minimalistic: nearly all interactions with PyBIDS
functionality occur through a core ``BIDSLayout`` object initialized by passing
in a path to a BIDS dataset. For most applications, no custom configuration
should be required.

Although technically still in alpha release, PyBIDS is already being used both
as a dependency in dozens of other open-source brain imaging packages --
*e.g.*, fMRIPrep [@fmriprep], MRIQC [@mriqc], datalad-neuroimaging
(https://github.com/datalad/datalad-neuroimaging), and fitlins
(https://github.com/poldracklab/fitlins) -- and directly in many researchers'
custom Python workflows. Development is extremely active, with bug fixes and
new features continually being added (https://github.com/bids-standard/pybids),
and major releases occurring approximately every 6 months. As of this writing,
29 people have contributed code to PyBIDS, and many more have provided feedback
and testing. The API is relatively stable, and documentation and testing
standards follow established norms for open-source scientific software. We
encourage members of the brain imaging community currently working in Python to
try using PyBIDS, and welcome new contributions.

# Acknowledgements

PyBIDS development is partly supported by NIH awards R01MH109682 (PI: Yarkoni),
R24MH114705 (PI: Poldrack), R01EB020740 (PI: Ghosh), and P41EB019936 (PI:
Kennedy), and NSF award 1429999 (PI: Halchenko).

# References
