import re

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
    >>> from bids.grabbids import BIDSValidator
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
        >>> from bids.grabbids import BIDSValidator
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
