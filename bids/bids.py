from __future__ import absolute_import, division, print_function

from .due import due, Doi

__all__ = []


# Use duecredit (duecredit.org) to provide a citation to relevant work to
# be cited. This does nothing, unless the user has duecredit installed,
# And calls this with duecredit (as in `python -m duecredit script.py`):
due.cite(Doi("10.1038/sdata.2016.44"),
         description="Brain Imaging Data Structure",
         tags=["reference-implementation"],
         path='bids')


