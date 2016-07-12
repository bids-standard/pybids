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


    def get_bold_run_filenames(self, subj, task, ses=None):
        if ses == None: ses = task
        sub=self.get_subj_foldernames()[subj-1]
        ses='ses-'+ses
        task='task-'+task
        bolds=[]
        for fil in os.listdir(_opj(self.basedir,sub,ses,'func')):
            if fil.endswith('bold.nii') or fil.endswith('bold.nii.gz'):
                bolds.append(fil)
        return bolds        


    def get_task_bold_run_filenames(self, task, ses=None):
        """
        """
        if ses == None: ses = task

        out = {}
        for sub in range(len(self.get_subj_foldernames())):
            runs = self.get_bold_run_filenames(sub, ses, task)
            if len(runs):
                out[sub+1] = runs
        return out

        def _parse_filename(fname):
            """ By Hanke
            """
            components = fname.split('_')
            ftype = components[-1]
            components = dict([c.split('-') for c in components[:-1]])
            components['filetype'] = ftype
            return components
 
        

basedir = '/Users/andrebeukers/Documents/fMRI/Python/Study Forrest/video_studyforrest'
bids = bidsdir(basedir)
bids.get_subj_foldernames
bids.get_bold_run_filenames(1,task='movie')
bids.get_task_bold_run_filenames(task='movie')


