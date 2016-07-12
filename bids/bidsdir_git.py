"""
Traverse main directory and harvest paths and file names
Based off of mvpa2.OpenFMRIDataset
"""

import os
from os.path import join as _opj

class bidsdir(object):

    def __init__(self, basedir):
        """
        """
        self.basedir = os.path.expanduser(os.path.expandvars(basedir))

    def get_subj_foldernames(self):
        """
        """

        subj_ids = []
        for fil in os.listdir(self.basedir):
            if fil.startswith('sub-'):
                subj_ids.append(fil)
        return subj_ids


    def get_bold_run_filenames(self, subj, ses, task):
        sub=self.get_subj_foldernames()[subj-1]
        ses='ses-'+ses
        task='task-'+task
        bolds=[]
        for fil in os.listdir(_opj(self.basedir,sub,ses,'func')):
            if fil.endswith('bold.nii') or fil.endswith('bold.nii.gz'):
                bolds.append(fil)
        return bolds        


    def get_task_bold_run_filenames(self, ses, task):
        """
        """
        out = {}
        for sub in range(len(self.get_subj_foldernames())):
            runs = self.get_bold_run_filenames(sub, ses, task)
            if len(runs):
                out[sub+1] = runs
        return out
        

basedir = '/Users/andrebeukers/Documents/fMRI/Python/Study Forrest/video_studyforrest'
bids = bidsdir(basedir)
bids.get_subj_foldernames
bids.get_bold_run_filenames(1,ses='movie',task='movie')
bids.get_task_bold_run_filenames(ses='movie',task='movie')

