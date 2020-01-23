"""
Contains helper functions that involve writing operations.
"""

import warnings
import os
import re
import sys
from string import Formatter
from itertools import product
from ..utils import splitext, listify
from os.path import join, dirname, exists, islink, isabs, isdir

__all__ = ['build_path', 'write_contents_to_file']

_PATTERN_FIND = re.compile(r'({([\w\d]*?)(?:<([^>]+)>)?(?:\|((?:\.?[\w])+))?\})')


def build_path(entities, path_patterns, strict=False):
    """
    Constructs a path given a set of entities and a list of potential
    filename patterns to use.

    Parameters
    ----------
    entities : :obj:`dict`
        A dictionary mapping entity names to entity values.
        Entities with ``None`` or empty-string value will be removed.
        Otherwise, entities will be cast to string values, therefore
        if any format is expected (e.g., zero-padded integers), the
        value should be formatted.
    path_patterns : :obj:`str` or :obj:`list`
        One or more filename patterns to write
        the file to. Entities should be represented by the name
        surrounded by curly braces. Optional portions of the patterns
        should be denoted by square brackets. Entities that require a
        specific value for the pattern to match can pass them inside
        angle brackets. Default values can be assigned by specifying a string after
        the pipe operator. E.g., (e.g., {type<image>|bold} would only match
        the pattern if the entity 'type' was passed and its value is
        "image", otherwise the default value "bold" will be used).
    strict : :obj:`bool`
        If True, all passed entities must be matched inside a
        pattern in order to be a valid match. If False, extra entities will
        be ignored so long as all mandatory entities are found.

    Returns
    -------
    A constructed path for this file based on the provided patterns, or
    ``None`` if no path was built given the combination of entities and patterns.

    Examples
    --------
    >>> entities = {
    ...     'extension': 'nii',
    ...     'space': 'MNI',
    ...     'subject': '001',
    ...     'suffix': 'inplaneT2',
    ... }
    >>> patterns = ['sub-{subject}[/ses-{session}]/anat/sub-{subject}[_ses-{session}]'
    ...             '[_acq-{acquisition}][_ce-{ceagent}][_rec-{reconstruction}]_'
    ...             '{suffix<T[12]w|T1rho|T[12]map|T2star|FLAIR|FLASH|PDmap|PD|PDT2|'
    ...             'inplaneT[12]|angio>}.{extension<nii|nii.gz|json>|nii.gz}',
    ...             'sub-{subject}[/ses-{session}]/anat/sub-{subject}[_ses-{session}]'
    ...             '[_acq-{acquisition}][_ce-{ceagent}][_rec-{reconstruction}]'
    ...             '[_space-{space}][_desc-{desc}]_{suffix<T1w|T2w|T1rho|T1map|T2map|'
    ...             'T2star|FLAIR|FLASH|PDmap|PD|PDT2|inplaneT[12]|angio>}.'
    ...             '{extension<nii|nii.gz|json>|nii.gz}']
    >>> build_path(entities, patterns)
    'sub-001/anat/sub-001_inplaneT2.nii'

    >>> build_path(entities, patterns, strict=True)
    'sub-001/anat/sub-001_space-MNI_inplaneT2.nii'

    >>> entities['space'] = None
    >>> build_path(entities, patterns, strict=True)
    'sub-001/anat/sub-001_inplaneT2.nii'

    >>> # If some entity is set to None, they are dropped
    >>> entities['extension'] = None
    >>> build_path(entities, patterns, strict=True)
    'sub-001/anat/sub-001_inplaneT2.nii.gz'

    >>> # If some entity is set to empty-string, they are dropped
    >>> entities['extension'] = ''
    >>> build_path(entities, patterns, strict=True)
    'sub-001/anat/sub-001_inplaneT2.nii.gz'

    >>> # If some selector is not in the pattern, skip it...
    >>> entities['datatype'] = 'anat'
    >>> build_path(entities, patterns)
    'sub-001/anat/sub-001_inplaneT2.nii.gz'

    >>> # ... unless the pattern should be strictly matched
    >>> entities['datatype'] = 'anat'
    >>> build_path(entities, patterns, strict=True) is None
    True

    >>> # If the value of an entity is not valid, do not match the pattern
    >>> entities['suffix'] = 'bold'
    >>> build_path(entities, patterns) is None
    True

    >>> entities = {
    ...     'extension': 'bvec',
    ...     'subject': '001',
    ... }
    >>> patterns = (
    ...     "sub-{subject}[/ses-{session}]/{datatype|dwi}/sub-{subject}[_ses-{session}]"
    ...     "[_acq-{acquisition}]_{suffix|dwi}.{extension<bval|bvec|json|nii.gz|nii>|nii.gz}"
    ... )
    >>> build_path(entities, patterns, strict=True)
    'sub-001/dwi/sub-001_dwi.bvec'

    >>> # Lists of entities are expanded
    >>> entities = {
    ...     'extension': 'bvec',
    ...     'subject': ['%02d' % i for i in range(1, 4)],
    ... }
    >>> build_path(entities, patterns, strict=True)
    ['sub-01/dwi/sub-01_dwi.bvec', 'sub-02/dwi/sub-02_dwi.bvec', 'sub-03/dwi/sub-03_dwi.bvec']

    """
    path_patterns = listify(path_patterns)

    # Drop None and empty-strings, keep zeros, and listify
    entities = {k: listify(v) for k, v in entities.items() if v or v == 0}

    # One less source of confusion
    if 'extension' in entities:
        entities['extension'] = [e.lstrip('.') for e in entities['extension']]

    # Loop over available patherns, return first one that matches all
    for pattern in path_patterns:
        entities_matched = list(_PATTERN_FIND.findall(pattern))
        defined = [e[1] for e in entities_matched]

        # If strict, all entities must be contained in the pattern
        if strict:
            if set(entities.keys()) - set(defined):
                continue

        # Iterate through the provided path patterns
        new_path = pattern

        # Expand options within valid values and
        # check whether entities provided have acceptable value
        tmp_entities = entities.copy()  # Do not modify the original query
        for fmt, name, valid, defval in entities_matched:
            valid_expanded = [v for val in valid.split('|') if val
                              for v in _expand_options(val)]
            if valid_expanded and defval and defval not in valid_expanded:
                warnings.warn(
                    'Pattern "%s" is inconsistent as it defines an invalid default value.' % fmt
                )
            if (
                valid_expanded
                and name in entities
                and set(entities[name]) - set(valid_expanded)
            ):
                continue

            if defval and name not in tmp_entities:
                tmp_entities[name] = [defval]

            # At this point, valid & default values are checked & set - simplify pattern
            new_path = new_path.replace(fmt, '{%s}' % name)

        optional_patterns = re.findall(r'(\[.*?\])', new_path)
        # Optional patterns with selector are cast to mandatory or removed
        for op in optional_patterns:
            for ent_name in {k for k, v in entities.items() if v is not None}:
                if ('{%s}' % ent_name) in op:
                    new_path = new_path.replace(op, op[1:-1])
                    continue

            # Surviving optional patterns are removed
            new_path = new_path.replace(op, '')

        # Replace entities
        fields = {pat[1] for pat in Formatter().parse(new_path)
                  if pat[1] and not pat[1].isdigit()}
        if fields - set(tmp_entities.keys()):
            continue

        tmp_entities = {k: v for k, v in tmp_entities.items()
                        if k in fields}

        new_path = [
            new_path.format(**e)
            for e in _expand_entities(tmp_entities)
        ]

        if new_path:
            if len(new_path) == 1:
                new_path = new_path[0]
            return new_path

    return None


def write_contents_to_file(path, contents=None, link_to=None,
                           content_mode='text', root=None, conflicts='fail'):
    """
    Uses provided filename patterns to write contents to a new path, given
    a corresponding entity map.

    Parameters
    ----------
    path : str
        Destination path of the desired contents.
    contents : str
        Raw text or binary encoded string of contents to write
        to the new path.
    link_to : str
        Optional path with which to create a symbolic link to.
        Used as an alternative to and takes priority over the contents
        argument.
    content_mode : {'text', 'binary'}
        Either 'text' or 'binary' to indicate the writing
        mode for the new file. Only relevant if contents is provided.
    root : str
        Optional root directory that all patterns are relative
        to. Defaults to current working directory.
    conflicts : {'fail', 'skip', 'overwrite', 'append'}
        One of 'fail', 'skip', 'overwrite', or 'append'
        that defines the desired action when the output path already
        exists. 'fail' raises an exception; 'skip' does nothing;
        'overwrite' overwrites the existing file; 'append' adds a suffix
        to each file copy, starting with 1. Default is 'fail'.
    """

    if root is None and not isabs(path):
        root = os.getcwd()

    if root:
        path = join(root, path)

    if exists(path) or islink(path):
        if conflicts == 'fail':
            msg = 'A file at path {} already exists.'
            raise ValueError(msg.format(path))
        elif conflicts == 'skip':
            msg = 'A file at path {} already exists, skipping writing file.'
            warnings.warn(msg.format(path))
            return
        elif conflicts == 'overwrite':
            if isdir(path):
                warnings.warn('New path is a directory, not going to '
                              'overwrite it, skipping instead.')
                return
            os.remove(path)
        elif conflicts == 'append':
            i = 1
            while i < sys.maxsize:
                path_splits = splitext(path)
                path_splits[0] = path_splits[0] + '_%d' % i
                appended_filename = os.extsep.join(path_splits)
                if not exists(appended_filename) and \
                        not islink(appended_filename):
                    path = appended_filename
                    break
                i += 1
        else:
            raise ValueError('Did not provide a valid conflicts parameter')

    if not exists(dirname(path)):
        os.makedirs(dirname(path))

    if link_to:
        os.symlink(link_to, path)
    elif contents:
        mode = 'wb' if content_mode == 'binary' else 'w'
        with open(path, mode) as f:
            f.write(contents)
    else:
        raise ValueError('One of contents or link_to must be provided.')


def _expand_options(value):
    """
    Expand optional substrings of valid entity values.

    Examples
    --------
    >>> _expand_options('[Jj]son[12]')
    ['Json1', 'Json2', 'json1', 'json2']

    >>> _expand_options('json')
    ['json']

    """
    expand_patterns = re.findall(r'\[(.*?)\]', value)
    if not expand_patterns:
        return [value]

    value = re.sub(r'\[(.*?)\]', '%s', value)
    return [value % _r for _r in product(*expand_patterns)]


def _expand_entities(entities):
    """
    Generate multiple replacement queries based on all combinations of values.

    Examples
    --------
    >>> entities = {'subject': ['01', '02'], 'session': ['1', '2'], 'task': ['rest', 'finger']}
    >>> out = _expand_entities(entities)
    >>> len(out)
    8

    >>> {'subject': '01', 'session': '1', 'task': 'rest'} in out
    True

    >>> {'subject': '02', 'session': '1', 'task': 'rest'} in out
    True

    >>> {'subject': '01', 'session': '2', 'task': 'rest'} in out
    True

    >>> {'subject': '02', 'session': '2', 'task': 'rest'} in out
    True

    >>> {'subject': '01', 'session': '1', 'task': 'finger'} in out
    True

    >>> {'subject': '02', 'session': '1', 'task': 'finger'} in out
    True

    >>> {'subject': '01', 'session': '2', 'task': 'finger'} in out
    True

    >>> {'subject': '02', 'session': '2', 'task': 'finger'} in out
    True

    """
    keys = list(entities.keys())
    values = list(product(*[entities[k] for k in keys]))
    return [{k: v for k, v in zip(keys, combs)} for combs in values]
