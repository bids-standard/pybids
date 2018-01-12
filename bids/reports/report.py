"""Generate publication-quality data acquisition methods section from BIDS dataset.
"""
from __future__ import print_function
import json
from os.path import dirname
from os.path import abspath
from os.path import join as pathjoin
from collections import Counter

from bids.reports import utils


class BIDSReport(object):
    """
    Generates publication-quality data acquisition methods section from BIDS
    dataset.
    """
    def __init__(self, layout, config=None):
        self.layout = layout
        if config is None:
            root = dirname(abspath(__file__))
            config = pathjoin(root, 'config', 'converters.json')
            with open(config) as fobj:
                self.config = json.load(fobj)

    def generate(self, task_converter=None):
        """Generate the methods section.
        """
        if task_converter is None:
            task_converter = {'rest': 'resting state'}

        self.config['task'] = task_converter

        descriptions = []

        subjs = self.layout.get_subjects()
        for sid in subjs:
            description = self._report(subj=sid)
            descriptions.append(description)
        counter = Counter(descriptions)
        print('Number of patterns detected: {0}'.format(len(counter.keys())))
        print(utils.warnings())
        print('')
        return counter

    def _report(self, subj):
        """Write a report.

        Parameters
        ----------
        layout : :obj:`bids.grabbids.BIDSLayout`
            Layout object for BIDS dataset.
        subj : :obj:`str`
            Subject ID.
        ses : :obj:`str`
            Session number.
        task_converter : :obj:`dict`, optional
            A dictionary converting task names as they appear in BIDS filenames to
            task names as the user would like them to appear in the report.
            Example: {'emotionalnback': 'emotional n-back task'}

        Returns
        -------
        description : :obj:`str`
            A publication-ready report of the dataset's data acquisition
            information. Each scan type is given its own paragraph.
        """

        description_list = []
        sessions = self.layout.get_sessions()
        if sessions:
            for ses in sessions:
                niftis = self.layout.get(subject=subj, session=ses, extensions='nii.gz')

                if niftis:
                    description_list.append('For session {0}:'.format(ses))
                    description_list += utils.parse_niftis(self.layout, niftis, subj, ses, self.config)
                    metadata = self.layout.get_metadata(niftis[0].filename)
        else:
            niftis = self.layout.get(subject=subj, extensions='nii.gz')

            if niftis:
                description_list += utils.parse_niftis(self.layout, niftis, subj, None, self.config)
                metadata = self.layout.get_metadata(niftis[0].filename)
            else:
                raise Exception('No niftis for subject {0}'.format(subj))

        # Assume all data were converted the same way and use the first nifti file's
        # json for conversion information.
        if 'metadata' not in vars():
            raise Exception('No valid jsons found. Cannot generate final paragraph.')

        #description_list.append(final_paragraph(metadata))
        #description_list = remove_duplicates(description_list)

        description = '\n\t'.join(description_list)
        description = description.replace('\tFor session', '\nFor session')
        description += '\n\n{0}'.format(utils.final_paragraph(metadata))
        return description
