"""Generate publication-quality data acquisition methods section from BIDS
dataset.

Parsing functions for generating the MRI data acquisition portion of a
methods section from a BIDS dataset.
"""
import logging
import os.path as op

import math
from num2words import num2words

from .. import __version__
from .utils import (num_to_str, list_to_str, get_slice_info,
                    get_seqstr, get_sizestr)

logging.basicConfig()
LOGGER = logging.getLogger('pybids.reports.parsing')


def general_acquisition_info(metadata):
    """
    General sentence on data acquisition. Should be first sentence in MRI data
    acquisition section.

    Parameters
    ----------
    metadata : :obj:`dict`
        The metadata for the dataset.

    Returns
    -------
    out_str : :obj:`str`
        Output string with scanner information.
    """
    out_str = ('MR data were acquired using a {tesla}-Tesla {manu} {model} '
               'MRI scanner.')
    out_str = out_str.format(tesla=metadata.get('MagneticFieldStrength',
                                                'UNKNOWN'),
                             manu=metadata.get('Manufacturer', 'MANUFACTURER'),
                             model=metadata.get('ManufacturersModelName',
                                                'MODEL'))
    return out_str


def get_slice_str(img, metadata):
    if 'SliceTiming' in metadata.keys():
        slice_order = ' in {0} order'.format(get_slice_info(metadata['SliceTiming']))
        n_slices = len(metadata['SliceTiming'])
    else:
        slice_order = ''
        n_slices = img.shape[2]
    slice_str = '{n_slices} slices{slice_order}'.format(
        n_slices=n_slices,
        slice_order=slice_order
    )
    return slice_str


def get_tr_str(metadata):
    tr = metadata['RepetitionTime'] * 1000
    tr = num_to_str(tr)
    tr_str = 'repetition time, TR={tr}ms'.format(tr=tr)
    return tr_str


def get_duration(img, metadata):
    tr = metadata['RepetitionTime']
    n_tps = img.shape[3]
    run_secs = math.ceil(n_tps * tr)
    mins, secs = divmod(run_secs, 60)
    duration = '{0}:{1:02.0f}'.format(int(mins), int(secs))
    return duration


def get_mbfactor_str(metadata):
    if metadata.get('MultibandAccelerationFactor', 1) > 1:
        mb_str = 'MB factor={}'.format(metadata['MultibandAccelerationFactor'])
    else:
        mb_str = ''
    return mb_str


def get_echotimes_str(metadata):
    """Build a description of echo times from metadata field.

    Parameters
    ----------
    metadata : dict
        Metadata information for multiple files merged into one dictionary.
        For multi-echo data, EchoTime should be a list.

    Returns
    -------
    te_str : str
        Description of echo times.
    me_str : str
        Whether the data are multi-echo or single-echo.
    """
    if 'EchoTime' in metadata.keys():
        if isinstance(metadata['EchoTime'], list):
            te = [num_to_str(t*1000) for t in metadata['EchoTime']]
            te = list_to_str(te)
            me_str = 'multi-echo'
        else:
            te = num_to_str(metadata['EchoTime']*1000)
            me_str = 'single-echo'
    else:
        te = 'UNKNOWN'
        me_str = 'UNKNOWN-echo'
    te_str = 'echo time, TE={}ms'.format(te)
    return te_str, me_str


def get_size_strs(img):
    """Build descriptions from sizes of imaging data, including field of view,
    voxel size, and matrix size.

    Parameters
    ----------
    img : nibabel.nifti1.Nifti1Image
        Image object from which to determine sizes.

    Returns
    -------
    fov_str
    matrixsize_str
    voxelsize_str
    """
    vs_str, ms_str, fov_str = get_sizestr(img)
    fov_str = 'field of view, FOV={}mm'.format(fov_str)
    voxelsize_str = 'voxel size={}mm'.format(vs_str)
    matrixsize_str = 'matrix size={}'.format(ms_str)
    return fov_str, matrixsize_str, voxelsize_str


def get_inplaneaccel_str(metadata):
    if metadata.get('ParallelReductionFactorInPlane', 1) > 1:
        pr_str = ('in-plane acceleration factor='
                  '{}'.format(metadata['ParallelReductionFactorInPlane']))
    else:
        pr_str = ''
    return pr_str


def get_flipangle_str(metadata):
    return 'flip angle, FA={}<deg>'.format(metadata.get('FlipAngle', 'UNKNOWN'))


def get_nvecs_str(img):
    return '{} diffusion directions'.format(img.shape[3])


def get_bval_str(bval_file):
    # Parse bval file
    with open(bval_file, 'r') as file_object:
        d = file_object.read().splitlines()
    bvals = [item for sublist in [l.split(' ') for l in d] for item in sublist]
    bvals = sorted([int(v) for v in set(bvals)])
    bvals = [num_to_str(v) for v in bvals]
    bval_str = list_to_str(bvals)
    bval_str = 'b-values of {} acquired'.format(bval_str)
    return bval_str


def get_dir_str(metadata, config):
    dir_str = config['dir'][metadata['PhaseEncodingDirection']]
    dir_str = 'phase encoding: {}'.format(dir_str)
    return dir_str


def get_for_str(metadata, layout):
    if 'IntendedFor' in metadata.keys():
        scans = metadata['IntendedFor']
        run_dict = {}
        for scan in scans:
            fn = op.basename(scan)
            if_file = [f for f in layout.get(extension=[".nii", ".nii.gz"]) if fn in f.path][0]
            run_num = int(if_file.run)
            target_type = if_file.entities['suffix'].upper()
            if target_type == 'BOLD':
                iff_meta = layout.get_metadata(if_file.path)
                task = iff_meta.get('TaskName', if_file.entities['task'])
                target_type_str = '{0} {1} scan'.format(task, target_type)
            else:
                target_type_str = '{0} scan'.format(target_type)

            if target_type_str not in run_dict.keys():
                run_dict[target_type_str] = []
            run_dict[target_type_str].append(run_num)

        for scan in run_dict.keys():
            run_dict[scan] = [num2words(r, ordinal=True) for r in sorted(run_dict[scan])]

        out_list = []
        for scan in run_dict.keys():
            if len(run_dict[scan]) > 1:
                s = 's'
            else:
                s = ''
            run_str = list_to_str(run_dict[scan])
            string = '{rs} run{s} of the {sc}'.format(
                rs=run_str, s=s, sc=scan)
            out_list.append(string)
        for_str = ' for the {0}'.format(list_to_str(out_list))
    else:
        for_str = ''
    return for_str


def clean_desc(desc):
    desc = desc.replace('\n', ' ').lstrip()
    while '  ' in desc:
        desc = desc.replace('  ', ' ')
    return desc


def func_info(layout, files, config):
    """
    Generate a paragraph describing T2*-weighted functional scans.

    Parameters
    ----------
    task : :obj:`str`
        The name of the task.
    n_runs : :obj:`int`
        The number of runs acquired for this task.
    metadata : :obj:`dict`
        The metadata for the scan from the json associated with the scan.
    img : :obj:`nibabel.Nifti1Image`
        Image corresponding to one of the runs.
    config : :obj:`dict`
        A dictionary with relevant information regarding sequences, sequence
        variants, phase encoding directions, and task names.

    Returns
    -------
    desc : :obj:`str`
        A description of the scan's acquisition information.
    """
    first_file = files[0]

    # General info
    task_name = metadata.get('TaskName', task+' task')
    seqs, variants = get_seqstr(config, metadata)
    if n_runs == 1:
        run_str = '{0} run'.format(num2words(n_runs).title())
    else:
        run_str = '{0} runs'.format(num2words(n_runs).title())

    # Parameters
    slice_str = get_slice_str(img, metadata)
    tr_str = get_tr_str(metadata)
    te_str, me_str = get_echotimes_str(metadata)
    fa_str = get_flipangle_str(metadata)
    fov_str, matrixsize_str, voxelsize_str = get_size_strs(img)
    mb_str = get_mbfactor_str(metadata)
    inplaneaccel_str = get_inplaneaccel_str(metadata)

    parameters_str = [slice_str, tr_str, te_str, fa_str,
                      fov_str, matrixsize_str, voxelsize_str,
                      mb_str, inplaneaccel_str]
    parameters_str = [d for d in parameters_str if len(d)]
    parameters_str = '; '.join(parameters_str)

    desc = '''
           {run_str} of {task} {variants} {seqs} {me_str} fMRI data were
           collected ({parameters_str}).
           Run duration was {duration} minutes, during which
           {n_vols} functional volumes were acquired.
           '''.format(run_str=run_str,
                      task=task_name,
                      variants=variants,
                      seqs=seqs,
                      me_str=me_str,
                      parameters_str=parameters_str,
                      duration=get_duration(img, metadata),
                      n_vols=img.shape[3],
                      )
    desc = clean_desc(desc)
    return desc


def anat_info(suffix, metadata, img, config):
    """
    Generate a paragraph describing T1- and T2-weighted structural scans.

    Parameters
    ----------
    suffix : :obj:`str`
        T1 or T2.
    metadata : :obj:`dict`
        Data from the json file associated with the scan, in dictionary
        form.
    img : :obj:`nibabel.Nifti1Image`
        The nifti image of the scan.
    config : :obj:`dict`
        A dictionary with relevant information regarding sequences, sequence
        variants, phase encoding directions, and task names.

    Returns
    -------
    desc : :obj:`str`
        A description of the scan's acquisition information.
    """
    # General info
    seqs, variants = get_seqstr(config, metadata)

    # Parameters
    slice_str = get_slice_str(img, metadata)
    tr_str = get_tr_str(metadata)
    te_str, me_str = get_echotimes_str(metadata)
    fa_str = get_flipangle_str(metadata)
    fov_str, matrixsize_str, voxelsize_str = get_size_strs(img)

    parameters_str = [slice_str, tr_str, te_str, fa_str,
                      fov_str, matrixsize_str, voxelsize_str]
    parameters_str = [d for d in parameters_str if len(d)]
    parameters_str = '; '.join(parameters_str)

    desc = '''
           {n_scans} {suffix} {variants} {seqs} {me_str} structural MRI scan(s)
           were collected ({parameters_str}).
           '''.format(n_scans='One',
                      suffix=suffix,
                      variants=variants,
                      seqs=seqs,
                      me_str=me_str,
                      parameters_str=parameters_str
                      )
    desc = clean_desc(desc)
    return desc


def dwi_info(bval_file, metadata, img, config):
    """
    Generate a paragraph describing DWI scan acquisition information.

    Parameters
    ----------
    bval_file : :obj:`str`
        File containing b-vals associated with DWI scan.
    metadata : :obj:`dict`
        Data from the json file associated with the DWI scan, in dictionary
        form.
    img : :obj:`nibabel.Nifti1Image`
        The nifti image of the DWI scan.
    config : :obj:`dict`
        A dictionary with relevant information regarding sequences, sequence
        variants, phase encoding directions, and task names.

    Returns
    -------
    desc : :obj:`str`
        A description of the DWI scan's acquisition information.
    """
    # General info
    seqs, variants = get_seqstr(config, metadata)

    # Parameters
    tr_str = get_tr_str(metadata)
    te_str, me_str = get_echotimes_str(metadata)
    fa_str = get_flipangle_str(metadata)
    fov_str, voxelsize_str, matrixsize_str = get_size_strs(img)
    bval_str = get_bval_str(bval_file)
    nvec_str = get_nvecs_str(img)
    mb_str = get_mbfactor_str(metadata)

    parameters_str = [slice_str, tr_str, te_str, fa_str, fov_str,
                      matrixsize_str, voxelsize_str,
                      bval_str, nvec_str,
                      mb_str]
    parameters_str = [d for d in parameters_str if len(d)]
    parameters_str = '; '.join(parameters_str)

    desc = '''
           One run of {variants} {seqs} diffusion-weighted (dMRI) data were
           collected ({parameters_str}).
           '''.format(variants=variants,
                      seqs=seqs,
                      parameters_str=parameters_str
                      )
    desc = clean_desc(desc)
    return desc


def fmap_info(metadata, img, config, layout):
    """
    Generate a paragraph describing field map acquisition information.

    Parameters
    ----------
    metadata : :obj:`dict`
        Data from the json file associated with the field map, in dictionary
        form.
    img : :obj:`nibabel.Nifti1Image`
        The nifti image of the field map.
    config : :obj:`dict`
        A dictionary with relevant information regarding sequences, sequence
        variants, phase encoding directions, and task names.

    Returns
    -------
    desc : :obj:`str`
        A description of the field map's acquisition information.
    """
    # General info
    seqs, variants = get_seqstr(config, metadata)

    # Parameters
    dir_str = get_dir_str(metadata, config)
    slice_str = get_slice_str(img, metadata)
    tr_str = get_tr_str(metadata)
    te_str, me_str = get_echotimes_str(metadata)
    fa_str = get_flipangle_str(metadata)
    fov_str, matrixsize_str, voxelsize_str = get_size_strs(img)
    mb_str = get_mbfactor_str(metadata)

    parameters_str = [dir_str, slice_str, tr_str, te_str, fa_str,
                      fov_str, matrixsize_str, voxelsize_str,
                      mb_str]
    parameters_str = [d for d in parameters_str if len(d)]
    parameters_str = '; '.join(parameters_str)

    for_str = get_for_str(metadata, layout)

    desc = '''
           A {variants} {seqs} field map ({parameters_str}) was
           acquired{for_str}.
           '''.format(variants=variants,
                      seqs=seqs,
                      for_str=for_str,
                      parameters_str=parameters_str
                      )
    desc = clean_desc(desc)
    return desc


def final_paragraph(metadata):
    """
    Describes dicom-to-nifti conversion process and methods generation.

    Parameters
    ----------
    metadata : :obj:`dict`
        The metadata for the scan.

    Returns
    -------
    desc : :obj:`str`
        Output string with scanner information.
    """
    if 'ConversionSoftware' in metadata.keys():
        soft = metadata['ConversionSoftware']
        vers = metadata['ConversionSoftwareVersion']
        software_str = ' using {soft} ({conv_vers})'.format(soft=soft, conv_vers=vers)
    else:
        software_str = ''
    desc = '''
           Dicoms were converted to NIfTI-1 format{software_str}.
           This section was (in part) generated
           automatically using pybids ({meth_vers}).
           '''.format(software_str=software_str,
                      meth_vers=__version__)
    desc = clean_desc(desc)
    return desc


def collect_associated_files(layout, files, extra_entities=()):
    """Collect and group BIDSFiles with multiple files per acquisition.

    Parameters
    ----------
    layout
    files : list of BIDSFile
    extra_entities

    Returns
    -------
    collected_files : list of list of BIDSFile
    """
    MULTICONTRAST_ENTITIES = ['echo', 'part', 'ch', 'direction']
    MULTICONTRAST_SUFFICES = [
        ('bold', 'phase'),
        ('phase1', 'phase2', 'phasediff', 'magnitude1', 'magnitude2'),
    ]
    if len(extra_entities):
        MULTICONTRAST_ENTITIES += extra_entities

    collected_files = []
    for f in files:
        if len(collected_files) and any(f in filegroup for filegroup in collected_files):
            continue
        ents = f.get_entities()
        ents = {k: v for k, v in ents.items() if k not in MULTICONTRAST_ENTITIES}

        # Group files with differing multi-contrast entity values, but same
        # everything else.
        all_suffices = ents['suffix']
        for mcs in MULTICONTRAST_SUFFICES:
            if ents['suffix'] in mcs:
                all_suffices = mcs
                break
        ents.pop('suffix')
        associated_files = layout.get(suffix=all_suffices, **ents)
        collected_files.append(associated_files)
    return collected_files


def parse_files(layout, data_files, sub, config, **kwargs):
    """
    Loop through files in a BIDSLayout and generate the appropriate description
    type for each scan. Compile all of the descriptions into a list.

    Parameters
    ----------
    layout : :obj:`bids.layout.BIDSLayout`
        Layout object for a BIDS dataset.
    data_files : :obj:`list` of :obj:`bids.layout.models.BIDSFile`
        List of nifti files in layout corresponding to subject/session combo.
    sub : :obj:`str`
        Subject ID.
    config : :obj:`dict`
        Configuration info for methods generation.
    """
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    # Group files into individual runs
    data_files = collect_associated_files(data_files, extra_entities=['run'])

    description_list = []
    # Assume all data have same basic info
    description_list.append(general_acquisition_info(data_files[0][0].get_metadata()))
    for group in data_files:
        if group[0].entities['datatype'] == 'func':
            group_description = func_info(layout, group, config)
        elif (group[0].entities['datatype'] == 'anat') and \
             data_file.entities['suffix'].endswith('w'):
            group_description = anat_info(layout, group, config)
        elif group[0].entities['datatype'] == 'dwi':
            group_description = dwi_info(layout, group, config)
        elif group[0].entities['datatype'] == 'fmap':
            group_description = fmap_info(layout, group, config)
        description_list.append(group_description)
    return description_list
