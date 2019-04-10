'''
Contains helper functions that involve writing operations.
'''

import warnings
import os
import re
import sys
from ..utils import splitext, listify
from os.path import join, dirname, exists, islink, isabs, isdir


__all__ = ['replace_entities', 'build_path', 'write_contents_to_file']


def replace_entities(entities, pattern):
    """
    Replaces all entity names in a given pattern with the corresponding
    values provided by entities.

    Args:
        entities (dict): A dictionary mapping entity names to entity values.
        pattern (str): A path pattern that contains entity names denoted
            by curly braces. Optional portions denoted by square braces.
            For example: 'sub-{subject}/[var-{name}/]{id}.csv'
            Accepted entity values, using regex matching, denoted within angle
            brackets.
            For example: 'sub-{subject<01|02>}/{task}.csv'

    Returns:
        A new string with the entity values inserted where entity names
        were denoted in the provided pattern.
    """
    ents = re.findall(r'\{(.*?)\}', pattern)
    new_path = pattern
    for ent in ents:
        match = re.search(r'([^|<]+)(<.*?>)?(\|.*)?', ent)
        if match is None:
            return None
        name, valid, default = match.groups()
        default = default[1:] if default is not None else default

        if name in entities and valid is not None:
            ent_val = str(entities[name])
            if not re.match(valid[1:-1], ent_val):
                if default is None:
                    return None
                entities[name] = default

        ent_val = entities.get(name, default)
        if ent_val is None:
            return None
        new_path = new_path.replace('{%s}' % ent, str(ent_val))

    return new_path


def build_path(entities, path_patterns, strict=False):
    """
    Constructs a path given a set of entities and a list of potential
    filename patterns to use.

    Args:
        entities (dict): A dictionary mapping entity names to entity values.
        path_patterns (str, list): One or more filename patterns to write
            the file to. Entities should be represented by the name
            surrounded by curly braces. Optional portions of the patterns
            should be denoted by square brackets. Entities that require a
            specific value for the pattern to match can pass them inside
            carets. Default values can be assigned by specifying a string after
            the pipe operator. E.g., (e.g., {type<image>|bold} would only match
            the pattern if the entity 'type' was passed and its value is
            "image", otherwise the default value "bold" will be used).
                Example 1: 'sub-{subject}/[var-{name}/]{id}.csv'
                Result 2: 'sub-01/var-SES/1045.csv'
        strict (bool): If True, all passed entities must be matched inside a
            pattern in order to be a valid match. If False, extra entities will
            be ignored so long as all mandatory entities are found.

    Returns:
        A constructed path for this file based on the provided patterns.
    """
    path_patterns = listify(path_patterns)

    # Loop over available patherns, return first one that matches all
    for pattern in path_patterns:
        # If strict, all entities must be contained in the pattern
        if strict:
            defined = re.findall(r'\{(.*?)(?:<[^>]+>)?\}', pattern)
            if set(entities.keys()) - set(defined):
                continue
        # Iterate through the provided path patterns
        new_path = pattern
        optional_patterns = re.findall(r'\[(.*?)\]', pattern)
        # First build from optional patterns if possible
        for optional_pattern in optional_patterns:
            optional_chunk = replace_entities(entities, optional_pattern) or ''
            new_path = new_path.replace('[%s]' % optional_pattern,
                                        optional_chunk)
        # Replace remaining entities
        new_path = replace_entities(entities, new_path)

        if new_path:
            return new_path

    return None


def write_contents_to_file(path, contents=None, link_to=None,
                           content_mode='text', root=None, conflicts='fail'):
    """
    Uses provided filename patterns to write contents to a new path, given
    a corresponding entity map.

    Args:
        path (str): Destination path of the desired contents.
        contents (str): Raw text or binary encoded string of contents to write
            to the new path.
        link_to (str): Optional path with which to create a symbolic link to.
            Used as an alternative to and takes priority over the contents
            argument.
        content_mode (str): Either 'text' or 'binary' to indicate the writing
            mode for the new file. Only relevant if contents is provided.
        root (str): Optional root directory that all patterns are relative
            to. Defaults to current working directory.
        conflicts (str): One of 'fail', 'skip', 'overwrite', or 'append'
            that defines the desired action when the output path already
            exists. 'fail' raises an exception; 'skip' does nothing;
            'overwrite' overwrites the existing file; 'append' adds  a suffix
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
