"""Generate publication-quality data acquisition methods section from BIDS dataset.

Utilities to generate the MRI data acquisition portion of a
methods section from a BIDS dataset.
"""
from __future__ import print_function
import re
from os.path import basename

import numpy as np
import nibabel as nib
from num2words import num2words

from bids.version import __version__


def warnings():
    """
    Remind users about things they need to do after generating the report.
    """
    return('Remember to double-check everything and to replace <deg> with '
           'a degree symbol.')


def remove_duplicates(seq):
    """
    Return unique elements from list while preserving order.
    From https://stackoverflow.com/a/480227/2589328
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def num_to_str(num):
    """
    Convert an int or float to a nice string.
    E.g.,
        21 -> '21'
        2.500 -> '2.5'
        3. -> '3'
    """
    return '{0:0.02f}'.format(num).rstrip('0').rstrip('.')


def list_to_str(lst):
    """
    Turn a list into a comma- and/or and-separated string.

    Parameters
    ----------
    lst : :obj:`list`
        A list of strings to join into a single string.

    Returns
    -------
    str_ : :obj:`str`
        A string with commas and/or ands separating th elements from ``lst``.

    """
    if len(lst) == 1:
        str_ = lst[0]
    elif len(lst) == 2:
        str_ = ' and '.join(lst)
    elif len(lst) > 2:
        str_ = ', '.join(lst[:-1])
        str_ += ', and {0}'.format(lst[-1])
    else:
        raise ValueError('List of length 0 provided.')
    return str_


def get_slice_info(slice_times):
    """
    Extract slice order from slice timing info.

    TODO: Be more specific with slice orders.
    Currently anything where there's some kind of skipping is interpreted as
    interleaved of some kind.

    Parameters
    ----------
    slice_times : array-like
        A list of slice times in seconds or milliseconds or whatever.

    Returns
    -------
    slice_order_name : :obj:`str`
        The name of the slice order sequence.
    """
    # Slice order
    slice_times = remove_duplicates(slice_times)
    slice_order = sorted(range(len(slice_times)), key=lambda k: slice_times[k])
    if slice_order == range(len(slice_order)):
        slice_order_name = 'sequential ascending'
    elif slice_order == reversed(range(len(slice_order))):
        slice_order_name = 'sequential descending'
    elif slice_order[0] < slice_order[1]:
        # We're allowing some wiggle room on interleaved.
        slice_order_name = 'interleaved ascending'
    elif slice_order[0] > slice_order[1]:
        slice_order_name = 'interleaved descending'
    else:
        slice_order = [str(s) for s in slice_order]
        raise Exception('Unknown slice order: [{0}]'.format(', '.join(slice_order)))

    return slice_order_name


def get_seqstr(config, metadata):
    """
    Extract and reformat imaging sequence(s) and variant(s) into pretty
    strings.

    Parameters
    ----------
    config : :obj:`dict`
        A dictionary with relevant information regarding sequences, sequence
        variants, phase encoding directions, and task names.
    metadata : :obj:`dict`
        The metadata for the scan.

    Returns
    -------
    seqs : :obj:`str`
        Sequence names.
    variants : :obj:`str`
        Sequence variant names.
    """
    seq_abbrs = metadata.get('ScanningSequence', '').split('_')
    seqs = [config['seq'].get(seq, seq) for seq in seq_abbrs]
    variants = [config['seqvar'].get(var, var) for var in \
                metadata.get('SequenceVariant', '').split('_')]
    seqs = list_to_str(seqs)
    if seq_abbrs[0]:
        seqs += ' ({0})'.format('/'.join(seq_abbrs))
    variants = list_to_str(variants)
    return seqs, variants


def get_sizestr(img):
    """
    Extract and reformat voxel size, matrix size, field of view, and number of
    slices into pretty strings.

    Parameters
    ----------
    img : :obj:`nibabel.Nifti1Image`
        Image from scan from which to derive parameters.

    Returns
    -------
    n_slices : :obj:`int`
        Number of slices.
    voxel_size : :obj:`str`
        Voxel size string (e.g., '2x2x2')
    matrix_size : :obj:`str`
        Matrix size string (e.g., '128x128')
    fov : :obj:`str`
        Field of view string (e.g., '256x256')
    """
    n_x, n_y, n_slices = img.shape[:3]
    voxel_dims = np.array(img.get_header().get_zooms()[:3])
    matrix_size = '{0}x{1}'.format(num_to_str(n_x), num_to_str(n_y))
    voxel_size = 'x'.join([num_to_str(s) for s in voxel_dims])
    fov = [n_x, n_y] * voxel_dims[:2]
    fov = 'x'.join([num_to_str(s) for s in fov])
    return n_slices, voxel_size, matrix_size, fov


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
    out_str = ('MR data were acquired using a {tesla}-Tesla {manu} {model} MRI '
               'scanner.')
    out_str = out_str.format(tesla=metadata['MagneticFieldStrength'],
                             manu=metadata['Manufacturer'],
                             model=metadata['ManufacturersModelName'])
    return out_str


def func_info(task, n_runs, metadata, img, config):
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
    if metadata.get('MultibandAccelerationFactor', 1) > 1:
        mb_str = '; MB factor={0}'.format(metadata['MultibandAccelerationFactor'])
    else:
        mb_str = ''

    if metadata.get('ParallelReductionFactorInPlane', 1) > 1:
        pr_str = '; in-plane acceleration factor={}'.format(metadata['ParallelReductionFactorInPlane'])
    else:
        pr_str = ''

    seqs, variants = get_seqstr(config, metadata)
    n_slices, vs_str, ms_str, fov_str = get_sizestr(img)

    tr = metadata['RepetitionTime']
    n_tps = img.shape[3]
    run_secs = np.ceil(n_tps * tr)
    mins, secs = divmod(run_secs, 60)
    length = '{0}:{1:02.0f}'.format(int(mins), int(secs))

    if n_runs == 1:
        run_str = '{0} run'.format(num2words(n_runs).title())
    else:
        run_str = '{0} runs'.format(num2words(n_runs).title())

    desc = '''
           {run_str} of {task} {variants} {seqs} fMRI data were collected
           ({n_slices} slices in {so} order; repetition time, TR={tr}ms;
           echo time, TE={te}ms; flip angle, FA={fa}<deg>;
           field of view, FOV={fov}mm; matrix size={ms};
           voxel size={vs}mm{mb_str}{pr_str}).
           Each run was {length} minutes in length, during which
           {n_vols} functional volumes were acquired.
           '''.format(run_str=run_str,
                      task=task,
                      variants=variants,
                      seqs=seqs,
                      n_slices=n_slices,
                      so=get_slice_info(metadata['SliceTiming']),
                      tr=num_to_str(tr*1000),
                      te=num_to_str(metadata['EchoTime']*1000),
                      fa=metadata['FlipAngle'],
                      vs=vs_str,
                      fov=fov_str,
                      ms=ms_str,
                      length=length,
                      n_vols=n_tps,
                      mb_str=mb_str,
                      pr_str=pr_str
                     )
    desc = desc.replace('\n', ' ').lstrip()
    while '  ' in desc:
        desc = desc.replace('  ', ' ')

    return desc


def anat_info(type_, metadata, img, config):
    """
    Generate a paragraph describing T1- and T2-weighted structural scans.

    Parameters
    ----------
    type_ : :obj:`str`
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
    n_slices, vs_str, ms_str, fov_str = get_sizestr(img)
    seqs, variants = get_seqstr(config, metadata)

    desc = '''
           {type_} {variants} {seqs} structural MRI data were collected
           ({n_slices} slices; repetition time, TR={tr}ms;
           echo time, TE={te}ms; flip angle, FA={fa}<deg>;
           field of view, FOV={fov}mm; matrix size={ms}; voxel size={vs}mm).
           '''.format(type_=type_,
                      variants=variants,
                      seqs=seqs,
                      n_slices=n_slices,
                      tr=num_to_str(metadata['RepetitionTime']*1000),
                      te=num_to_str(metadata['EchoTime']*1000),
                      fa=metadata['FlipAngle'],
                      vs=vs_str,
                      fov=fov_str,
                      ms=ms_str,
                     )
    desc = desc.replace('\n', ' ').lstrip()
    while '  ' in desc:
        desc = desc.replace('  ', ' ')

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
    # Parse bval file
    with open(bval_file, 'r') as file_object:
        d = file_object.read().splitlines()
    bvals = [item for sublist in [l.split(' ') for l in d] for item in sublist]
    bvals = sorted([int(v) for v in set(bvals)])
    bvals = [str(v) for v in bvals]
    if len(bvals) == 1:
        bval_str = bvals[0]
    elif len(bvals) == 2:
        bval_str = ' and '.join(bvals)
    else:
        bval_str = ', '.join(bvals[:-1])
        bval_str += ', and {0}'.format(bvals[-1])

    if metadata.get('MultibandAccelerationFactor', 1) > 1:
        mb_str = '; MB factor={0}'.format(metadata['MultibandAccelerationFactor'])
    else:
        mb_str = ''

    n_slices, vs_str, ms_str, fov_str = get_sizestr(img)
    n_vecs = img.shape[3]
    seqs, variants = get_seqstr(config, metadata)
    variants = variants[0].upper() + variants[1:]  # variants starts sentence

    desc = '''
           {variants} {seqs} diffusion-weighted (dMRI) data were collected
           ({n_slices} slices in {so} order; repetition time, TR={tr}ms;
           echo time, TE={te}ms; flip angle, FA={fa}<deg>;
           field of view, FOV={fov}mm; matrix size={ms}; voxel size={vs}mm;
           b-values of {bval_str} acquired;
           {n_vecs} diffusion directions{mb_str}).
           '''.format(variants=variants,
                      seqs=seqs,
                      n_slices=n_slices,
                      so=get_slice_info(metadata['SliceTiming']),
                      tr=num_to_str(metadata['RepetitionTime']*1000),
                      te=num_to_str(metadata['EchoTime']*1000),
                      fa=metadata['FlipAngle'],
                      vs=vs_str,
                      fov=fov_str,
                      ms=ms_str,
                      bval_str=bval_str,
                      n_vecs=n_vecs,
                      mb_str=mb_str
                     )
    desc = desc.replace('\n', ' ').lstrip()
    while '  ' in desc:
        desc = desc.replace('  ', ' ')

    return desc


def fmap_info(metadata, img, config):
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
    dir_ = config['dir'][metadata['PhaseEncodingDirection']]
    n_slices, vs_str, ms_str, fov_str = get_sizestr(img)
    seqs, variants = get_seqstr(config, metadata)

    if 'IntendedFor' in metadata.keys():
        scans = metadata['IntendedFor']
        run_dict = {}
        for scan in scans:
            fn = basename(scan)
            run_search = re.search(r'.*_run-([0-9]+).*', fn)
            run_num = int(run_search.groups()[0])
            type_search = re.search(r'.*_([a-z0-9]+)\..*', fn)
            ty = type_search.groups()[0].upper()
            if ty == 'BOLD':
                task_search = re.search(r'.*_task-([a-z0-9]+).*', fn)
                task = config['task'].get(task_search.groups()[0],
                                          task_search.groups()[0])
                ty_str = '{0} {1} scan'.format(task, ty)
            else:
                ty_str = '{0} scan'.format(ty)

            if ty_str not in run_dict.keys():
                run_dict[ty_str] = []
            run_dict[ty_str].append(run_num)

        for scan in run_dict.keys():
            run_dict[scan] = [num2words(r, ordinal=True) for r in sorted(run_dict[scan])]

        out_list = []
        for scan in run_dict.keys():
            if len(run_dict[scan]) > 1:
                s = 's'
            else:
                s = ''
            run_str = list_to_str(run_dict[scan])
            string = '{rs} run{s} of the {sc}'.format(rs=run_str,
                                                      s=s,
                                                      sc=scan)
            out_list.append(string)
        for_str = ' for the {0}'.format(list_to_str(out_list))
    else:
        for_str = ''

    desc = '''
           A {variants} {seqs} field map (phase encoding:
           {dir_}; {n_slices} slices; repetition time, TR={tr}ms;
           echo time, TE={te}ms; flip angle, FA={fa}<deg>;
           field of view, FOV={fov}mm; matrix size={ms};
           voxel size={vs}mm) was acquired{for_str}.
           '''.format(variants=variants,
                      seqs=seqs,
                      dir_=dir_,
                      for_str=for_str,
                      n_slices=n_slices,
                      tr=num_to_str(metadata['RepetitionTime']*1000),
                      te=num_to_str(metadata['EchoTime']*1000),
                      fa=metadata['FlipAngle'],
                      vs=vs_str,
                      fov=fov_str,
                      ms=ms_str)
    desc = desc.replace('\n', ' ').lstrip()
    while '  ' in desc:
        desc = desc.replace('  ', ' ')

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
        software_str = ' using {soft} ({conv_vers})'.format(soft=metadata['ConversionSoftware'],
                                                            conv_vers=metadata['ConversionSoftwareVersion'])
    else:
        software_str = ''
    desc = '''
           Dicoms were converted to NIfTI-1 format{software_str}.
           This section was (in part) generated
           automatically using pybids ({meth_vers}).
           '''.format(software_str=software_str,
                      meth_vers=__version__)
    desc = desc.replace('\n', ' ').lstrip()
    while '  ' in desc:
        desc = desc.replace('  ', ' ')

    return desc


def parse_niftis(layout, niftis, subj, ses, config):
    """
    Loop through niftis in a BIDSLayout and generate the appropriate description
    type for each scan. Compile all of the descriptions into a list.

    Parameters
    ----------
    layout : :obj:`bids.grabbids.BIDSLayout`
        Layout object for a BIDS dataset.
    niftis : :obj:`list` or :obj:`grabbit.core.File`
        List of nifti files in layout corresponding to subject/session combo.
    subj : :obj:`str`
        Subject ID.
    ses : :obj:`str` or :obj:`None`
        Session number. Can be None.
    config : :obj:`dict`
        Configuration info for methods generation.
    """
    description_list = []
    skip_task = {}
    for nifti_struct in niftis:
        nii_file = nifti_struct.filename
        metadata = layout.get_metadata(nii_file)
        if not metadata:
            print('No json file found for {0}'.format(nii_file))
        else:
            img = nib.load(nii_file)

            # Assume all data were acquired the same way.
            if not description_list:
                description_list.append(general_acquisition_info(metadata))

            if nifti_struct.modality == 'func':
                task = config['task'].get(nifti_struct.task, nifti_struct.task+' task')
                if not skip_task.get(task, False):
                    if ses:
                        n_runs = len(layout.get(subject=subj, session=ses,
                                                extensions='nii.gz', task=nifti_struct.task))
                    else:
                        n_runs = len(layout.get(subject=subj, extensions='nii.gz',
                                                task=nifti_struct.task))
                    description_list.append(func_info(task, n_runs, metadata, img, config))
                    skip_task[task] = True

            elif nifti_struct.modality == 'anat':
                type_ = nifti_struct.type
                if type_.endswith('w'):
                    type_ = type_[:-1] + '-weighted'
                description_list.append(anat_info(type_, metadata, img, config))
            elif nifti_struct.modality == 'dwi':
                bval_file = nii_file.replace('.nii.gz', '.bval')
                description_list.append(dwi_info(bval_file, metadata, img, config))
            elif nifti_struct.modality == 'fmap':
                description_list.append(fmap_info(metadata, img, config))

    return description_list
