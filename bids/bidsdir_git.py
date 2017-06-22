"""
Traverse main directory and harvest paths and flee names
Inspired by mvpa2.OpenFMRIDataset

"""

import os
from os.path import join as _opj

def _parse_filename(fname):
    """ by Hanke 
        input BIDS filename
        returns dict with file info"""
    components_list = fname.split('_')
    components_dict = dict([c.split('-') for c in components_list[:-1]])
    components_dict['filetype'] = components_list[-1]
    return components_dict

def _assemble_bold_filepath(fname):
    """ input bold_filename
        returns bids path from basedir to func folder """
    
    file_info = _parse_filename(fname)

    if 'ses' in file_info.keys(): 
        return _opj('sub-' + file_info['sub'], 'ses-' + file_info['ses'], 'func')
    else: 
        return _opj('sub-' + file_info['sub'], 'func')



class bidsdir(object):

    def __init__(self, basedir):
        """
        """
        self.basedir = os.path.expanduser(os.path.expandvars(basedir))

    def get_subjfoldernames(self):
        """ Bids object method 
            returns name of folders containing subject data """
        subj_ids = []
        for fle in os.listdir(self.basedir):
            if fle.startswith('sub-'):
                subj_ids.append(fle)
        return subj_ids


    def get_subj_boldfilenames(self, subj, ses=''):
        """ input subject number, session name (if more than one session)
            returns name of bold.nii files """  

        if len(ses)>0: ses = 'ses-' + ses 

        sub = self.get_subjfoldernames()[subj-1]
        bolds=[]
        for fle in os.listdir(_opj(self.basedir,sub,ses,'func')):            
            if fle.endswith('bold.nii') or fle.endswith('bold.nii.gz'):
                bolds.append(fle)

        return bolds
         
    def get_subj_boldpaths(self, subj, ses=''):
        """ input subject number, and optionally sessionid 
            returns path to all bold files of given subject"""

        if len(self.get_subj_boldfilenames(subj, ses)):
            sample_filename=self.get_subj_boldfilenames(subj, ses)[0]
            # use sample_filename to find path to func folder
            path2funcfolder=_opj(self.basedir, _assemble_bold_filepath(sample_filename))
        else: return []        

        bold = []
        for fle in os.listdir(path2funcfolder):
            if fle.endswith(('_bold.nii','_bold.nii.gz')):
                bold.append(_opj(path2funcfolder,fle))
        return bold


    def get_subj_taskids(self, subj, ses=''):
        """ input subject number, optionally session id 
            returns name of task runs """
        tasks=[]
        for run_filename in self.get_subj_boldfilenames(subj,ses):
            tsk=_parse_filename(run_filename)['task']
            tasks.append(tsk)
        return tasks

    def get_task_boldfilenames(self, task, ses=''):
        """ input taskid, optionally sessionid
            returns name of bold files for given subject """        
        
        task_bolds=[]
        for sub in range(len(self.get_subjfoldernames())):
            sub_bolds=self.get_subj_boldfilenames(sub+1,ses)
            if task in self.get_subj_taskids(sub):
                idx = self.get_subj_taskids(sub).index(task)
                task_bolds.append(self.get_subj_boldfilenames(sub)[idx])
        return task_bolds

    def get_task_boldpaths(self, task, ses=''):
        """ input task id, optionally sessionid
            returns path to to all files with task id """
        task_bolds=[]
        for fle in self.get_task_boldfilenames(task=task,ses=ses):
            task_bolds.append(_opj(self.basedir,_assemble_bold_filepath(fle)))
        return task_bolds





 
        
""" For testing """

basedir = '/Users/andrebeukers/Documents/fMRI/Python/dataset/raw_data'
#basedir = '/Users/andrebeukers/Documents/fMRI/Python/Study Forrest/video_studyforrest'
bids = bidsdir(basedir)
#bids.get_subjfoldernames()
#bids.get_subj_boldfilenames(subj=3)
#bids.get_subject_taskids(subj=3)
#bids.get_task_boldfilenames(task='pa')
bids.get_task_boldpaths(task='pa')



