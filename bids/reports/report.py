"""Generate publication-quality data acquisition methods section from BIDS dataset.
"""
from collections import Counter

from bids.reports import utils


class BIDSReport(object):
    """
    Generates publication-quality data acquisition methods section from BIDS
    dataset.
    """
    def __init__(self, layout):
        self.layout = layout

    def generate(self, task_converter=None):
        """Generate the methods section.
        """
        if task_converter is None:
            task_converter = {}
        descriptions = []
        sessions = self.layout.get_sessions()
        for ses in sessions:
            subjs = self.layout.get_subjects(session=ses)
            for sid in subjs:
                description = utils.report(self.layout, subj=sid, ses=ses,
                                           task_converter=task_converter)
                descriptions.append(description)
        counter = Counter(descriptions)
        print('Number of patterns detected: {0}'.format(len(counter.keys())))
        return counter
