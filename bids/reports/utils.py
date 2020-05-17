"""Generate publication-quality data acquisition methods section from BIDS dataset.

Utilities to generate the MRI data acquisition portion of a
methods section from a BIDS dataset.
"""
import logging

logging.basicConfig()
LOGGER = logging.getLogger('pybids.reports.utils')


def reminder():
    """
    Remind users about things they need to do after generating the report.
    """
    return('Remember to double-check everything and to replace <deg> with '
           'a degree symbol.')


def remove_duplicates(seq):
    """
    Return unique elements from list while preserving order.
    From https://stackoverflow.com/a/480227/2589328
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def num_to_str(num):
    """
    Convert an int or float to a nice string.
    E.g.,
        21 -> '21'
        2.500 -> '2.5'
        3. -> '3'
    """
    return '{0:0.02f}'.format(num).rstrip('0').rstrip('.')


def list_to_str(lst):
    """
    Turn a list into a comma- and/or and-separated string.

    Parameters
    ----------
    lst : :obj:`list`
        A list of strings to join into a single string.

    Returns
    -------
    str_ : :obj:`str`
        A string with commas and/or ands separating the elements from ``lst``.

    """
    if len(lst) == 1:
        str_ = lst[0]
    elif len(lst) == 2:
        str_ = ' and '.join(lst)
    elif len(lst) > 2:
        str_ = ', '.join(lst[:-1])
        str_ += ', and {0}'.format(lst[-1])
    else:
        raise ValueError('List of length 0 provided.')
    return str_


def clean_multiline(desc):
    """
    Remove newlines and double-spaces from multiline string.
    """
    desc = desc.replace('\n', ' ').lstrip()
    while '  ' in desc:
        desc = desc.replace('  ', ' ')
    return desc


def collect_associated_files(layout, files, extra_entities=()):
    """Collect and group BIDSFiles with multiple files per acquisition.

    Parameters
    ----------
    layout
    files : list of BIDSFile
    extra_entities

    Returns
    -------
    collected_files : list of list of BIDSFile
    """
    MULTICONTRAST_ENTITIES = ['echo', 'part', 'ch', 'direction']
    MULTICONTRAST_SUFFICES = [
        ('bold', 'phase'),
        ('phase1', 'phase2', 'phasediff', 'magnitude1', 'magnitude2'),
    ]
    if len(extra_entities):
        MULTICONTRAST_ENTITIES += extra_entities

    collected_files = []
    for f in files:
        if len(collected_files) and any(f in filegroup for filegroup in collected_files):
            continue
        ents = f.get_entities()
        ents = {k: v for k, v in ents.items() if k not in MULTICONTRAST_ENTITIES}

        # Group files with differing multi-contrast entity values, but same
        # everything else.
        all_suffices = ents['suffix']
        for mcs in MULTICONTRAST_SUFFICES:
            if ents['suffix'] in mcs:
                all_suffices = mcs
                break
        ents.pop('suffix')
        associated_files = layout.get(suffix=all_suffices, **ents)
        collected_files.append(associated_files)
    return collected_files
