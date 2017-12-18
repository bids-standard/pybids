"""Generate publication-quality data acquisition methods section from BIDS dataset.
"""
import json
from os.path import join, abspath, splitext

import nibabel as nib

from bids.grabbids import BIDSLayout
from bids.reports import utils


def report(bids_dir, subj, ses, task_converter):
    """Write a report.
    """
    layout = BIDSLayout(bids_dir)

    # Remove potential trailing slash with abspath
    subj_dir = abspath(join(bids_dir, subj))

    # Get json files for field maps
    jsons = layout.get(subject=subj, session=ses, extensions='json')

    description_list = []
    for json_struct in jsons:
        json_file = json_struct.filename
        nii_file = splitext(json_file)[0] + '.nii.gz'
        with open(json_file, 'r') as file_object:
            json_data = json.load(file_object)
        img = nib.load(nii_file)

        # Assume all data were acquired the same way.
        if not description_list:
            description_list.append(utils.general_acquisition_info(json_data))

        if json_struct.modality == 'func':
            task = task_converter.get(json_struct.task, json_struct.task)
            n_runs = len(layout.get(subject=subj, session=ses,
                                    extensions='json', task=json_struct.task))
            description_list.append(utils.func_info(task, n_runs, json_data, img))
        elif json_struct.modality == 'anat':
            type_ = json_struct.type[:-1]
            description_list.append(utils.anat_info(type_, json_data, img))
        elif json_struct.modality == 'dwi':
            bval_file = splitext(json_file)[0] + '.bval'
            description_list.append(utils.dwi_info(bval_file, json_data, img))
        elif json_struct.modality == 'fmap':
            description_list.append(utils.fmap_info(json_data, img, task_converter, subj_dir))

    # Assume all data were converted the same way.
    description_list.append(utils.final_paragraph(json_data))
    description_list = utils.remove_duplicates(description_list)
    description = '\n\n'.join(description_list)
    return description
