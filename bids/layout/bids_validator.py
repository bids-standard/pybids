import re
import hashlib
import pandas as pd
import json

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
        self.anat_suffixes = ["T1w", "T2w", "T1map", "T2map",
                              "T1rho", "FLAIR", "PD", "PDT2",
                              "inplaneT1", "inplaneT2", "angio",
                              "defacemask", "SWImagandphase"]
        self.index_associated = index_associated

    def is_bids(self, path):
        """Checks if a file path appropriate for BIDS.

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

        return (
            self.is_top_level(path) |
            self.is_associated_data(path) |
            self.is_session_level(path) |
            self.is_subject_level(path) |
            self.is_anat(path) |
            self.is_dwi(path) |
            self.is_func(path) |
            self.is_behavioral(path) |
            self.is_cont(path) |
            self.is_field_map(path) |
            self.is_phenotypic(path)
        )

    def is_top_level(self, path):
        ''' Check if the file has appropriate name for a top-level file. '''
        fixed_top_level_names = ["/README", "/CHANGES", "/dataset_description.json", "/participants.tsv",
                                 "/participants.json", "/phasediff.json", "/phase1.json", "/phase2.json", "/fieldmap.json"]

        func_top_re = re.compile('^\\/(?:ses-[a-zA-Z0-9]+_)?(?:recording-[a-zA-Z0-9]+_)?task-[a-zA-Z0-9]+(?:_acq-[a-zA-Z0-9]+)?(?:_rec-[a-zA-Z0-9]+)?(?:_run-[0-9]+)?(?:_echo-[0-9]+)?'
                                 + '(_bold.json|_sbref.json|_events.json|_events.tsv|_physio.json|_stim.json|_beh.json)$')
        func_top_flag = False if func_top_re.search(path) is None else True

        anat_top_re = re.compile('^\\/(?:ses-[a-zA-Z0-9]+_)?(?:_acq-[a-zA-Z0-9]+)?(?:_rec-[a-zA-Z0-9]+)?(?:_run-[0-9]+_)?'
                                 + '(' + "|".join(self.anat_suffixes) + ').json$')
        anat_top_flag = False if anat_top_re.search(path) is None else True

        dwi_top_re = re.compile('^\\/(?:ses-[a-zA-Z0-9]+)?(?:_acq-[a-zA-Z0-9]+)?(?:_rec-[a-zA-Z0-9]+)?(?:_run-[0-9]+)?(?:_)?'
                                + 'dwi.(?:json|bval|bvec)$')
        dwi_top_flag = False if dwi_top_re.search(path) is None else True

        multi_dir_fieldmap_re = re.compile(
            '^\\/(?:dir-[a-zA-Z0-9]+)_epi.json$')
        multi_dir_fieldmap_flag = False if multi_dir_fieldmap_re.search(
            path) is None else True

        other_top_files_re = re.compile('^\\/(?:ses-[a-zA-Z0-9]+_)?(?:recording-[a-zA-Z0-9]+)?(?:task-[a-zA-Z0-9]+_)?(?:_acq-[a-zA-Z0-9]+)?(?:_rec-[a-zA-Z0-9]+)?(?:_run-[0-9]+)?'
                                        + '(_physio.json|_stim.json)$')
        other_top_files_flag = False if other_top_files_re.search(
            path) is None else True

        check_index = path in fixed_top_level_names
        return (check_index |
                func_top_flag | anat_top_flag |
                dwi_top_flag | multi_dir_fieldmap_flag | other_top_files_flag)

    def is_associated_data(self, path):
        ''' Check if file is appropriate associated data. '''
        if not self.index_associated:
            return False
        associated_data_re = re.compile(
            '^\\/(?:code|derivatives|sourcedata|stimuli)\\/(?:.*)$')
        associated_data_flag = associated_data_re.search(path)
        return associated_data_flag is not None

    def is_phenotypic(self, path):
        ''' Check if file is phenotypic data. '''
        phenotypic_data = re.compile('^\\/(?:phenotype)\\/(?:.*.tsv|.*.json)$')
        return phenotypic_data.search(path) is not None

    def is_session_level(self, path):
        ''' Check if the file has appropriate name for a session level. '''
        scans_re = re.compile('^\\/(sub-[a-zA-Z0-9]+)' +
                              '\\/(?:(ses-[a-zA-Z0-9]+)' +
                              '\\/)?\\1(_\\2)?(_scans.tsv|_scans.json)$')

        func_ses_re = re.compile('^\\/(sub-[a-zA-Z0-9]+)' +
                                 '\\/(?:(ses-[a-zA-Z0-9]+)' +
                                 '\\/)?\\1(_\\2)?task-[a-zA-Z0-9]+(?:_acq-[a-zA-Z0-9]+)?(?:_rec-[a-zA-Z0-9]+)?(?:_run-[0-9]+)?(?:_echo-[0-9]+)?'
                                 + '(_bold.json|_sbref.json|_events.json|_events.tsv|_physio.json|_stim.json)$')

        anat_ses_re = re.compile('^\\/(sub-[a-zA-Z0-9]+)' +
                                 '\\/(?:(ses-[a-zA-Z0-9]+)' +
                                 '\\/)?\\1(_\\2)?(?:_acq-[a-zA-Z0-9]+)?(?:_rec-[a-zA-Z0-9]+)?(?:_run-[0-9]+_)?'
                                 + '(' + ("|").join(self.anat_suffixes) + ').json$')

        dwi_ses_re = re.compile('^\\/(sub-[a-zA-Z0-9]+)' +
                                '\\/(?:(ses-[a-zA-Z0-9]+)' +
                                '\\/)?\\1(_\\2)?(?:_acq-[a-zA-Z0-9]+)?(?:_rec-[a-zA-Z0-9]+)?(?:_run-[0-9]+)?(?:_)?'
                                + 'dwi.(?:json|bval|bvec)$')

        return (self.conditional_match(scans_re, path) |
                self.conditional_match(func_ses_re, path) |
                self.conditional_match(anat_ses_re, path) |
                self.conditional_match(dwi_ses_re, path))

    def is_subject_level(self, path):
        ''' Check if the file has appropriate name for a subject level. '''
        scans_re = re.compile('^\\/(sub-[a-zA-Z0-9]+)' +
                              '\\/\\1(_sessions.tsv|_sessions.json)$')
        return scans_re.search(path) is not None

    def is_anat(self, path):
        ''' Check if the file has a name appropriate for an anatomical scan.
        '''
        anat_re = re.compile('^\\/(sub-[a-zA-Z0-9]+)' +
                             '\\/(?:(ses-[a-zA-Z0-9]+)' +
                             '\\/)?anat' +
                             '\\/\\1(_\\2)?(?:_acq-[a-zA-Z0-9]+)?(?:_rec-[a-zA-Z0-9]+)?(?:_run-[0-9]+)?_(?:'
                             + "|".join(self.anat_suffixes)
                             + ').(nii.gz|nii|json)$')
        return self.conditional_match(anat_re, path)

    def is_dwi(self, path):
        ''' Check if the file has a name appropriate for a diffusion scan. '''
        suffixes = ["dwi", "sbref"]
        anat_re = re.compile('^\\/(sub-[a-zA-Z0-9]+)' +
                             '\\/(?:(ses-[a-zA-Z0-9]+)' +
                             '\\/)?dwi' +
                             '\\/\\1(_\\2)?(?:_acq-[a-zA-Z0-9]+)?(?:_rec-[a-zA-Z0-9]+)?(?:_run-[0-9]+)?_(?:'
                             + ("|").join(suffixes)
                             + ').(nii.gz|nii|json|bvec|bval)$')
        return self.conditional_match(anat_re, path)

    def is_field_map(self, path):
        ''' Check if the file has a name appropriate for a fieldmap scan. '''
        suffixes = ["phasediff", "phase1", "phase2", "magnitude1",
                    "magnitude2", "magnitude", "fieldmap", "epi"]
        anat_re = re.compile('^\\/(sub-[a-zA-Z0-9]+)' +
                             '\\/(?:(ses-[a-zA-Z0-9]+)' +
                             '\\/)?fmap' +
                             '\\/\\1(_\\2)?(?:_acq-[a-zA-Z0-9]+)?(?:_rec-[a-zA-Z0-9]+)?(?:_dir-[a-zA-Z0-9]+)?(?:_run-[0-9]+)?_(?:'
                             + ("|").join(suffixes)
                             + ').(nii.gz|nii|json)$')
        return self.conditional_match(anat_re, path)

    def is_func(self, path):
        ''' Check if the file has a name appropriate for a functional scan. '''
        func_re = re.compile('^\\/(sub-[a-zA-Z0-9]+)' +
                             '\\/(?:(ses-[a-zA-Z0-9]+)' +
                             '\\/)?func' +
                             '\\/\\1(_\\2)?_task-[a-zA-Z0-9]+(?:_acq-[a-zA-Z0-9]+)?(?:_rec-[a-zA-Z0-9]+)?(?:_run-[0-9]+)?(?:_echo-[0-9]+)?'
                             + '(?:_bold.nii.gz|_bold.nii|_bold.json|_sbref.nii.gz|_sbref.json|_events.json|_events.tsv|_physio.tsv.gz|_stim.tsv.gz|_physio.json|_stim.json|_defacemask.nii.gz|_defacemask.nii)$')
        return self.conditional_match(func_re, path)

    def is_behavioral(self, path):
        ''' Check if the file has a name appropriate for behavioral data. '''
        func_beh = re.compile('^\\/(sub-[a-zA-Z0-9]+)' +
                              '\\/(?:(ses-[a-zA-Z0-9]+)' +
                              '\\/)?beh' +
                              '\\/\\1(_\\2)?_task-[a-zA-Z0-9]+(?:_acq-[a-zA-Z0-9]+)?(?:_rec-[a-zA-Z0-9]+)?(?:_run-[0-9]+)?'
                              + '(?:_beh.json|_events.json|_events.tsv|_physio.tsv.gz|_stim.tsv.gz|_physio.json|_stim.json)$')
        return self.conditional_match(func_beh, path)

    def is_func_bold(self, path):
        ''' Check if the file has a name appropriate for functional bold. '''
        func_re = re.compile('^\\/(sub-[a-zA-Z0-9]+)' +
                             '\\/(?:(ses-[a-zA-Z0-9]+)' +
                             '\\/)?func' +
                             '\\/\\1(_\\2)?_task-[a-zA-Z0-9]+(?:_acq-[a-zA-Z0-9]+)?(?:_rec-[a-zA-Z0-9]+)?' +
                             '(?:_run-[0-9]+)?(?:_echo-[0-9]+)?'
                             + '(?:_bold.nii.gz|_bold.nii|_sbref.nii.gz|_sbref.nii)$')
        return self.conditional_match(func_re, path)

    def is_cont(self, path):
        ''' Check if the file has a name appropriate for physiological and
        continuous recordings. '''
        cont_re = re.compile('^\\/(sub-[a-zA-Z0-9]+)' +
                             '\\/(?:(ses-[a-zA-Z0-9]+)' +
                             '\\/)?(?:func|beh)' +
                             '\\/\\1(_\\2)?_task-[a-zA-Z0-9]+(?:_acq-[a-zA-Z0-9]+)?(?:_rec-[a-zA-Z0-9]+)?(?:_run-[0-9]+)?' +
                             '(?:_recording-[a-zA-Z0-9]+)?'
                             + '(?:_physio.tsv.gz|_stim.tsv.gz|_physio.json|_stim.json)$')
        return self.conditional_match(cont_re, path)

    def get_path_values(self, path):
        ''' Takes a file path and returns values found for the following path
        keys:
            sub-
            ses-
        '''
        values = {}
        # capture subject
        match = re.compile(r'/^\/sub-([a-zA-Z0-9]+)/)').findall(path)
        values['sub'] = match[1] if match & match[1] else None

        # capture session
        match = re.compile(
            r'/^\/sub-[a-zA-Z0-9]+\/ses-([a-zA-Z0-9]+)/').findall(path)
        values['ses'] = match[1] if match & match[1] else None

        return values

    def conditional_match(self, expression, path):
        match = expression.findall(path)
        match = match[0] if len(match) >= 1 else False
        # adapted from JS code and JS does not support conditional groups
        if (match):
            if ((match[1] == match[2][1:]) | (not match[1])):
                return True
            else:
                return False
        else:
            return False


def validate_sequences(layout, config):
    """Checks files in BIDS project match user defined expectations.

    This method is a wrapper for the duplicate_check() and 
    expected_file_check() methods. Use it to check whether there are
    files with duplicate content within the BIDS data set and to check
    the number of data set files against a user customized configuration
    file. Returns three data frames: duplicate_file_df, summary_df, problem_df.


    Parameters
    ----------
        layout: BIDSLayout class
            A BIDSLayout path of a data set.

        config: string
            Path to customized configuration file.


    Examples
    --------
    >>> layout = bids.grabbids.BIDSLayout('/path/to/sample/project_root')
    >>> validate_sequences(layout, '/path/to/sample/config.json')
    >>> duplicate_file_df
    # Put example output here
    >>> summary_df
    # Put example output here
    >>> problem_df
    # Put example output here
    """
    
    duplicate_file_df = duplicate_check(layout)
    summary_df, problem_df = expected_file_check(layout, config)
    return duplicate_file_df, summary_df, problem_df
    
    
def duplicate_check(layout):
    """Checks images in BIDS project are not duplicated.

    Check whether any files have duplicate content within the 
    BIDS data set. Returns a data frame: duplicate_file_df.


    Parameters
    ----------
        layout: BIDSLayout class
            A BIDSLayout path of a data set.


    Examples
    --------
    >>> layout = bids.grabbids.BIDSLayout('/path/to/sample/project_root')
    >>> validate_sequences(layout, '/path/to/sample/config.json')
    >>> duplicate_file_df
    # Put example output here
    """
    
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
    out_df = df.stack().reset_index().drop(columns='level_1').rename(columns={'level_0': 'hash', 0: 'filename'}) 
    return out_df
    
    
def expected_file_check(layout, config):
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
    >>> layout = bids.grabbids.BIDSLayout('/path/to/sample/project_root')
    >>> validate_sequences(layout, '/path/to/sample/config.json')
    >>> summary_df
    # Put example output here
    >>> problem_df
    # Put example output here


    Notes
    --------
    The configuration file must contain the following keys: 
        "modality"
        "session"
        "type"
        "runs"
    
    Additional keys matching the inputs for `layout.get` may also be added 
    (e.g. 'task' for functionals).
    Values shoud match those in the BIDS file names. See the sample config 
    for an example (bids/layout/tests/data/sample_validation_config.json)

    """

    summary_df = pd.DataFrame(columns=['subject', 'session', 'modality', 'task', 'runs', 'runs_found', 'problem'])
    with open(config) as f:
        json_data = json.load(f)
        subjects = layout.get_subjects()
    for sub in subjects: 
        for scan_params_d in json_data['sequences']:
            scan_params = dict.copy(scan_params_d)
            seq_params = {i: scan_params[i] for i in scan_params if i != 'runs'}
            actual_runs = layout.get(return_type='obj', subject=sub, extensions='.nii.gz', **seq_params)
            scan_params['subject'] = sub
            scan_params['runs_found'] = len(actual_runs)
            if len(actual_runs) != scan_params['runs']:
                scan_params['problem'] = True
            else:
                scan_params['problem'] = False
            summary_df = summary_df.append(scan_params, ignore_index=True)
        print('sub {:}: done'.format(sub))
    problem_df = summary_df.loc[summary_df['problem'] == True]
    return summary_df, problem_df