"""Tools for validating BIDS projects."""

import re
import json
from os.path import join, abspath, dirname
from collections import namedtuple


__all__ = ['BIDSValidator']


class BIDSValidator():
    """An object for BIDS (Brain Imaging Data Structure) verification in a data.

    The main method of this class is `is_bids()`. You should use it for
    checking whether a file path compatible with BIDS.

    Parameters
    ----------
    index_associated : bool, default: True
        Specifies if an associated data should be checked. If it is true then
        any file paths in directories `code/`, `derivatives/`, `sourcedata/`
        and `stimuli/` will pass the validation, else they won't.

    Examples
    --------
    >>> from bids.layout import BIDSValidator
    >>> validator = BIDSValidator()
    >>> filepaths = ["/sub-01/anat/sub-01_rec-CSD_T1w.nii.gz",
    >>> "/sub-01/anat/sub-01_acq-23_rec-CSD_T1w.exe", #wrong extension
    >>> "/participants.tsv"]
    >>> for filepath in filepaths:
    >>>     print( validator.is_bids(filepath) )
    True
    False
    True
    """

    def __init__(self, index_associated=True):
        self.rule_dir = join(dirname(abspath(__file__)),'config', 'validator')
        self.index_associated = index_associated

    def is_bids(self, path):
        """Check if a file path appropriate for BIDS.

        Main method of the validator. uses other class methods for checking
        different aspects of the file path.

        Parameters
        ----------
            path: string
                A path of a file you want to check.

        Examples
        --------
        >>> from bids.layout import BIDSValidator
        >>> validator = BIDSValidator()
        >>> validator.is_bids("/sub-01/ses-test/anat/sub-01_ses-test_rec-CSD_run-23_T1w.nii.gz")
        True
        >>> validator.is_bids("/sub-01/ses-test/sub-01_run-01_dwi.bvec") # missed session in the filename
        False
        """
        conditions = []

        conditions.append(self.is_top_level(path))
        conditions.append(self.is_associated_data(path))
        conditions.append(self.is_session_level(path))
        conditions.append(self.is_subject_level(path))
        conditions.append(self.is_phenotypic(path))
        conditions.append(self.is_file(path))

        return (any(conditions))

    def is_top_level(self, path):
        """Check if the file has appropriate name for a top-level file."""
        with open(join(self.rule_dir, 'fixed_top_level_names.json'), 'r') as f:
            fixed_top_level_json = json.load(f)
            fixed_top_level_names = fixed_top_level_json['fixed_top_level_names']

        regexps = self.get_regular_expressions('top_level_rules.json')

        conditions = [False if re.compile(x).search(path) is None else True
                      for x in regexps]

        conditions.append(path in fixed_top_level_names)

        return (any(conditions))

    def is_associated_data(self, path):
        """Check if file is appropriate associated data."""
        if not self.index_associated:
            return False

        regexps = self.get_regular_expressions('associated_data_rules.json')

        conditions = [(re.compile(x).search(path) is not None)
                      for x in regexps]

        return any(conditions)

    def is_session_level(self, path):
        """Check if the file has appropriate name for a session level."""
        regexps = self.get_regular_expressions('session_level_rules.json')

        conditions = [self.conditional_match(x, path) for x in regexps]

        return (any(conditions))

    def is_subject_level(self, path):
        """Check if the file has appropriate name for a subject level."""
        regexps = self.get_regular_expressions('subject_level_rules.json')

        conditions = [(re.compile(x).search(path) is not None)
                      for x in regexps]

        return (any(conditions))

    def is_phenotypic(self, path):
        """Check if file is phenotypic data."""
        regexps = self.get_regular_expressions('phenotypic_rules.json')

        conditions = [(re.compile(x).search(path) is not None)
                      for x in regexps]

        return (any(conditions))

    def is_file(self, path):
        """Check if file is phenotypic data."""
        regexps = self.get_regular_expressions('file_level_rules.json')

        conditions = [(re.compile(x).search(path) is not None)
                      for x in regexps]

        return (any(conditions))

    def get_regular_expressions(self, filename):
        """Get regular expressions from file."""
        regexps = []

        filename = join(self.rule_dir, filename)

        with open(filename, 'r') as f:
            rules = json.load(f)

        for key in list(rules.keys()):
            rule = rules[key]

            regexp = rule["regexp"]

            if "tokens" in rule:
                tokens = rule["tokens"]

                for token in list(tokens):
                    regexp = regexp.replace(token, "|".join(tokens[token]))

            regexps.append(regexp)

        return regexps

    def conditional_match(self, expression, path):
        match = re.compile(expression).findall(path)
        match = match[0] if len(match) >= 1 else False
        # adapted from JS code and JS does not support conditional groups
        if (match):
            return ((match[1] == match[2][1:]) | (not match[1]))
        else:
            return False

def validate_sequences(layout, config):
    """Checks files in BIDS project match user defined expectations.

    This method is a wrapper for the check_duplicate_files() and 
    check_expected_files() methods. Use it to check whether there are
    files with duplicate content within the BIDS data set and to check
    the number of data set files against a user customized configuration
    file. Returns a named tuple of three data frames: duplicates, summary, and problems.


    Parameters
    ----------
        layout: BIDSLayout class
            A BIDSLayout path of a data set.

        config: string
            Path to customized configuration file. Requires `runs` as an input.
            See the sample config for an example (bids/layout/tests/data/sample_validation_config.json).


    Examples
    --------
    >>> layout = bids.grabbids.BIDSLayout('/path_to/sample_project_root')
    >>> dfs = validate_sequences(layout, 'pybids/bids/layout/tests/data/sample_validation_config.json')
    >>> dfs.duplicates
    # Put example output here
    >>> df.summary
    # Put example output here
    >>> df.problems
    # Put example output here
    """
    
    duplicate_file_df = check_duplicate_files(layout)
    summary_df, problem_df = check_expected_files(layout, config)
    output = namedtuple('output', ['duplicates', 'summary', 'problems'])
    return output(duplicate_file_df, summary_df, problem_df)
    
    
def check_duplicate_files(layout):
    """Checks images in BIDS project are not duplicated.

    Check whether any files have duplicate content within the 
    BIDS data set. Returns a data frame: duplicate_file_df.


    Parameters
    ----------
        layout: BIDSLayout class
            A BIDSLayout path of a data set.


    Examples
    --------
    >>> layout = bids.grabbids.BIDSLayout('/path_to/sample_project_root')
    >>> duplicate_file_df = check_duplicate_files(layout)
    >>> duplicate_file_df
    # Put example output here


    Notes
    ------
    Returns a data frame in which the first column is the file
    identifier and the second column is the path to the file.
    Files with matching identifiers have the same content.
    """
    
    import pandas as pd
    import hashlib
    def md5(fname):
        hash_md5 = hashlib.md5()
        with open(fname, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    hash_map = {}
    all_niftis = layout.get(return_type="file", extensions='.nii.gz')
    for nifti_file in all_niftis:
        md5sum = md5(nifti_file)
        if md5sum in hash_map:
            hash_map[md5sum].append(nifti_file)
        else:
            hash_map[md5sum] = [nifti_file]
    df = pd.DataFrame.from_dict(hash_map, orient='index') 
    pruned_df = df.stack().reset_index().drop(columns='level_1')
    out_df = pruned_df.rename(columns={'level_0': 'hash', 0: 'filename'})
    return out_df
    
    
def check_expected_files(layout, config):
    """Checks files in BIDS project match user defined expectations.

    This method checks the number of data set files against a user customized 
    configuration file. Returns two data frames: summary_df, problem_df.


    Parameters
    ----------
        layout: BIDSLayout class
            A BIDSLayout path of a data set.

        config: string
            Path to customized configuration file.


    Examples
    --------
    >>> layout = bids.grabbids.BIDSLayout('/path_to/sample_project_root')
    >>> summary_df, problem_df = check_expected_files(layout, 'pybids/bids/layout/tests/data/sample_validation_config.json')
    >>> summary_df
    # Put example output here
    >>> problem_df
    # Put example output here


    Notes
    --------

    `runs` is a mandatory field in the config file.
    
    The configuration file can take any keys that are valid arguments for
    pybids `layout.get()` Values shoud match those in the BIDS file names. 
    See the sample config for an example (bids/layout/tests/data/sample_validation_config.json).
    The more specific keys are provided, the more informative the output will be.

    """

    import pandas as pd
    dictlist = []
    with open(config) as f:
        json_data = json.load(f)
        subjects = layout.get_subjects()
    for sub in subjects: 
        for scan_params_d in json_data['sequences']:
            scan_params = scan_params_d.copy()
            seq_params = {i: scan_params[i] for i in scan_params if i != 'runs'}
            actual_runs = layout.get(return_type='obj', subject=sub, extensions='.nii.gz', **seq_params)
            scan_params['subject'] = sub
            scan_params['runs_found'] = len(actual_runs)
            scan_params['problem'] = len(actual_runs) != scan_params['runs']
            dictlist.append(scan_params)
    summary_df = pd.DataFrame(dictlist)
    problem_df = summary_df.loc[summary_df['problem'] == True]
    return summary_df, problem_df