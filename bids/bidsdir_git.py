"""
Traverse main directory and harvest paths and file names
Based off of mvpa2.OpenFMRIDataset
"""

import os
from os.path import join as _opj

def _parse_filename(fname):
    """ By Hanke
    """
    components = fname.split('_')
    ftype = components[-1]
    components = dict([c.split('-') for c in components[:-1]])
    components['filetype'] = ftype
    return components


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


    def get_bold_run_filenames(self, subj, ses=''):
        
        sub=self.get_subj_foldernames()[subj-1]
        if len(ses)>0: ses = 'ses-' + ses   

        bolds=[]
        for fil in os.listdir(_opj(self.basedir,sub,ses,'func')):            
            if fil.endswith('bold.nii') or fil.endswith('bold.nii.gz'):
                bolds.append(fil)
        return bolds

    def get_tasks(self, subj, ses=''):
        tasks=[]
        for run_filename in self.get_bold_run_filenames(subj,ses):
            tsk=_parse_filename(run_filename)['task']
            tasks.append(tsk)
        return tasks


    def get_task_bold_run_filenames(self, task='', ses=''):
        """
        """
        
        out = {}
        for sub in range(len(self.get_subj_foldernames())):
            runs = self.get_bold_run_filenames(sub, ses)
            if len(runs):
                out[sub+1] = runs
        return out


 
        


""" For testing """

basedir = '/Users/andrebeukers/Documents/fMRI/Python/dataset/raw_data'
bids = bidsdir(basedir)
# bids.get_subj_foldernames()
# bids.get_bold_run_filenames(subj=3)
bids.get_tasks(subj=3)
# bids.get_task_bold_run_filenames()

# sample_fname = bids.get_bold_run_filenames(1,'movie')[0]
# _parse_filename(sample_fname)
