from __future__ import absolute_import, division, print_function
from .version import __version__  # noqa
from .due import due, Doi

__all__ = ["grabbids", "analysis", "reports"]

due.cite(Doi("10.1038/sdata.2016.44"),
         description="Brain Imaging Data Structure",
         tags=["reference-implementation"],
         path='bids')
