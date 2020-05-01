"""Generate publication-quality data acquisition methods section from BIDS
dataset.
"""
import json
from os.path import dirname
from os.path import abspath
from os.path import join as pathjoin
from collections import Counter

from bids.reports import utils
from bids.reports import parsing


class BIDSReport(object):
    """
    Generates publication-quality data acquisition methods section from BIDS
    dataset.

    Parameters
    ----------
    layout : :obj:`bids.layout.BIDSLayout`
        Layout object for a BIDS dataset.
    config : :obj:`str` or :obj:`dict`, optional
        Configuration info for methods generation. Can be a path to a file
        (str), a dictionary, or None. If None, loads and uses default
        configuration information.
        Keys in the dictionary include:
            'dir':      a dictionary for converting encoding direction strings
                        (e.g., j-) to descriptions (e.g., anterior to
                        posterior)
            'seq':      a dictionary of sequence abbreviations (e.g., EP) and
                        corresponding names (e.g., echo planar)
            'seqvar':   a dictionary of sequence variant abbreviations
                        (e.g., SP) and corresponding names (e.g., spoiled)
    """
    def __init__(self, layout, config=None):
        self.layout = layout
        if config is None:
            config = pathjoin(dirname(abspath(__file__)), 'config',
                              'converters.json')

        if isinstance(config, str):
            with open(config) as fobj:
                config = json.load(fobj)

        if not isinstance(config, dict):
            raise ValueError('Input config must be None, dict, or path to '
                             'json file containing dict.')

        self.config = config

    def generate(self, **kwargs):
        r"""Generate the methods section.

        Parameters
        ----------
        task_converter : :obj:`dict`, optional
            A dictionary with information for converting task names from BIDS
            filename format to human-readable strings.

        Returns
        -------
        counter : :obj:`collections.Counter`
            A dictionary of unique descriptions across subjects in the dataset,
            along with the number of times each pattern occurred. In cases
            where all subjects underwent the same protocol, the most common
            pattern is most likely the most complete. In cases where the
            dataset contains multiple protocols, each pattern will need to be
            inspected manually.

        Examples
        --------
        >>> from os.path import join
        >>> from bids.layout import BIDSLayout
        >>> from bids.reports import BIDSReport
        >>> from bids.tests import get_test_data_path
        >>> layout = BIDSLayout(join(get_test_data_path(), 'synthetic'))
        >>> report = BIDSReport(layout)
        >>> counter = report.generate(session='01')
        Number of patterns detected: 1
        Remember to double-check everything and to replace <deg> with a degree symbol.

        >>> counter.most_common()[0][0]  # doctest: +ELLIPSIS
        'For session 01:\n\tMR data were...'

        """
        descriptions = []

        subjs = self.layout.get_subjects(**kwargs)
        kwargs = {k: v for k, v in kwargs.items() if k != 'subject'}
        for sid in subjs:
            descriptions.append(self._report_subject(subject=sid, **kwargs))
        counter = Counter(descriptions)
        print('Number of patterns detected: {0}'.format(len(counter.keys())))
        print(utils.reminder())
        return counter

    def _report_subject(self, subject, **kwargs):
        """Write a report for a single subject.

        Parameters
        ----------
        subject : :obj:`str`
            Subject ID.

        Attributes
        ----------
        layout : :obj:`bids.layout.BIDSLayout`
            Layout object for a BIDS dataset.
        config : :obj:`dict`
            Configuration info for methods generation.

        Returns
        -------
        description : :obj:`str`
            A publication-ready report of the dataset's data acquisition
            information. Each scan type is given its own paragraph.
        """
        description_list = []
        # Remove sess from kwargs if provided, else set sess as all available
        sessions = kwargs.pop('session',
                              self.layout.get_sessions(subject=subject,
                                                       **kwargs))
        if not sessions:
            sessions = [None]
        elif not isinstance(sessions, list):
            sessions = [sessions]

        for ses in sessions:
            niftis = self.layout.get(
                subject=subject, extension=[".nii", ".nii.gz"],
                **kwargs)

            if niftis:
                description_list.append('For session {0}:'.format(ses))
                description_list += parsing.parse_niftis(self.layout, niftis,
                                                         subject, self.config,
                                                         session=ses)
                metadata = self.layout.get_metadata(niftis[0].path)
            else:
                raise Exception('No niftis for subject {0}'.format(subject))

        # Assume all data were converted the same way and use the last nifti
        # file's json for conversion information.
        if 'metadata' not in vars():
            raise Exception('No valid jsons found. Cannot generate final '
                            'paragraph.')

        description = '\n\t'.join(description_list)
        description = description.replace('\tFor session', '\nFor session')
        description += '\n\n{0}'.format(parsing.final_paragraph(metadata))
        return description
