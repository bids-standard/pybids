"""Parsing functions for generating BIDSReports."""
import logging
import warnings

import nibabel as nib
from num2words import num2words

from .. import __version__
from ..utils import collect_associated_files
from . import parameters

logging.basicConfig()
LOGGER = logging.getLogger("pybids.reports.parsing")


def func_info(layout, files, config):
    """Generate a paragraph describing T2*-weighted functional scans.

    Parameters
    ----------
    layout : :obj:`bids.layout.BIDSLayout`
        Layout object for a BIDS dataset.
    files : :obj:`list` of :obj:`bids.layout.models.BIDSFile`
        List of nifti files in layout corresponding to DWI scan.
    config : :obj:`dict`
        A dictionary with relevant information regarding sequences, sequence
        variants, phase encoding directions, and task names.

    Returns
    -------
    desc : :obj:`str`
        A description of the scan's acquisition information.
    """
    first_file = files[0]
    metadata = first_file.get_metadata()
    img = nib.load(first_file.path)

    # General info
    task_name = first_file.get_entities()["task"] + " task"
    task_name = metadata.get("TaskName", task_name)
    seqs, variants = parameters.describe_sequence(metadata, config)
    all_runs = sorted(list(set([f.get_entities().get("run", 1) for f in files])))
    n_runs = len(all_runs)
    if n_runs == 1:
        run_str = "{0} run".format(num2words(n_runs).title())
    else:
        run_str = "{0} runs".format(num2words(n_runs).title())
    dur_str = parameters.describe_duration(files)

    # Parameters
    slice_str = parameters.describe_slice_timing(img, metadata)
    tr_str = parameters.describe_repetition_time(metadata)
    te_str, me_str = parameters.describe_echo_times(files)
    fa_str = parameters.describe_flip_angle(metadata)
    fov_str, matrixsize_str, voxelsize_str = parameters.describe_image_size(img)
    mb_str = parameters.describe_multiband_factor(metadata)
    inplaneaccel_str = parameters.describe_inplane_accel(metadata)

    parameters_str = [
        slice_str,
        tr_str,
        te_str,
        fa_str,
        fov_str,
        matrixsize_str,
        voxelsize_str,
        mb_str,
        inplaneaccel_str,
    ]
    parameters_str = [d for d in parameters_str if len(d)]
    parameters_str = "; ".join(parameters_str)

    desc = (
        "{run_str} of {task} {variants} {seqs} {me_str} fMRI data were "
        "collected ({parameters_str}). {dur_str}".format(
            run_str=run_str,
            task=task_name,
            variants=variants,
            seqs=seqs,
            me_str=me_str,
            parameters_str=parameters_str,
            dur_str=dur_str,
        )
    )
    return desc


def anat_info(layout, files, config):
    """Generate a paragraph describing T1- and T2-weighted structural scans.

    Parameters
    ----------
    layout : :obj:`bids.layout.BIDSLayout`
        Layout object for a BIDS dataset.
    files : :obj:`list` of :obj:`bids.layout.models.BIDSFile`
        List of nifti files in layout corresponding to DWI scan.
    config : :obj:`dict`
        A dictionary with relevant information regarding sequences, sequence
        variants, phase encoding directions, and task names.

    Returns
    -------
    desc : :obj:`str`
        A description of the scan's acquisition information.
    """
    first_file = files[0]
    metadata = first_file.get_metadata()
    img = nib.load(first_file.path)

    # General info
    seqs, variants = parameters.describe_sequence(metadata, config)
    all_runs = sorted(list(set([f.get_entities().get("run", 1) for f in files])))
    n_runs = len(all_runs)
    if n_runs == 1:
        run_str = "{0} run".format(num2words(n_runs).title())
    else:
        run_str = "{0} runs".format(num2words(n_runs).title())
    scan_type = first_file.get_entities()["suffix"].replace("w", "-weighted")

    # Parameters
    slice_str = parameters.describe_slice_timing(img, metadata)
    tr_str = parameters.describe_repetition_time(metadata)
    te_str, me_str = parameters.describe_echo_times(files)
    fa_str = parameters.describe_flip_angle(metadata)
    fov_str, matrixsize_str, voxelsize_str = parameters.describe_image_size(img)

    parameters_str = [
        slice_str,
        tr_str,
        te_str,
        fa_str,
        fov_str,
        matrixsize_str,
        voxelsize_str,
    ]
    parameters_str = [d for d in parameters_str if len(d)]
    parameters_str = "; ".join(parameters_str)

    desc = (
        "{run_str} of {scan_type} {variants} {seqs} {me_str} structural MRI "
        "data were collected ({parameters_str}).".format(
            run_str=run_str,
            scan_type=scan_type,
            variants=variants,
            seqs=seqs,
            me_str=me_str,
            parameters_str=parameters_str,
        )
    )
    return desc


def dwi_info(layout, files, config):
    """Generate a paragraph describing DWI scan acquisition information.

    Parameters
    ----------
    layout : :obj:`bids.layout.BIDSLayout`
        Layout object for a BIDS dataset.
    files : :obj:`list` of :obj:`bids.layout.models.BIDSFile`
        List of nifti files in layout corresponding to DWI scan.
    config : :obj:`dict`
        A dictionary with relevant information regarding sequences, sequence
        variants, phase encoding directions, and task names.

    Returns
    -------
    desc : :obj:`str`
        A description of the DWI scan's acquisition information.
    """
    first_file = files[0]
    metadata = first_file.get_metadata()
    img = nib.load(first_file.path)
    bval_file = first_file.path.replace(".nii.gz", ".bval").replace(".nii", ".bval")

    # General info
    seqs, variants = parameters.describe_sequence(metadata, config)
    all_runs = sorted(list(set([f.get_entities().get("run", 1) for f in files])))
    n_runs = len(all_runs)
    if n_runs == 1:
        run_str = "{0} run".format(num2words(n_runs).title())
    else:
        run_str = "{0} runs".format(num2words(n_runs).title())

    # Parameters
    tr_str = parameters.describe_repetition_time(metadata)
    te_str, me_str = parameters.describe_echo_times(files)
    fa_str = parameters.describe_flip_angle(metadata)
    fov_str, voxelsize_str, matrixsize_str = parameters.describe_image_size(img)
    bval_str = parameters.describe_bvals(bval_file)
    nvec_str = parameters.describe_dmri_directions(img)
    mb_str = parameters.describe_multiband_factor(metadata)

    parameters_str = [
        tr_str,
        te_str,
        fa_str,
        fov_str,
        matrixsize_str,
        voxelsize_str,
        bval_str,
        nvec_str,
        mb_str,
    ]
    parameters_str = [d for d in parameters_str if len(d)]
    parameters_str = "; ".join(parameters_str)

    desc = (
        "{run_str} of {variants} {seqs} diffusion-weighted (dMRI) data were "
        "collected ({parameters_str}).".format(
            run_str=run_str,
            variants=variants,
            seqs=seqs,
            parameters_str=parameters_str,
        )
    )
    return desc


def fmap_info(layout, files, config):
    """Generate a paragraph describing field map acquisition information.

    Parameters
    ----------
    layout : :obj:`bids.layout.BIDSLayout`
        Layout object for a BIDS dataset.
    files : :obj:`list` of :obj:`bids.layout.models.BIDSFile`
        List of nifti files in layout corresponding to field map scan.
    config : :obj:`dict`
        A dictionary with relevant information regarding sequences, sequence
        variants, phase encoding directions, and task names.

    Returns
    -------
    desc : :obj:`str`
        A description of the field map's acquisition information.
    """
    first_file = files[0]
    metadata = first_file.get_metadata()
    img = nib.load(first_file.path)

    # General info
    seqs, variants = parameters.describe_sequence(metadata, config)

    # Parameters
    dir_str = parameters.describe_pe_direction(metadata, config)
    slice_str = parameters.describe_slice_timing(img, metadata)
    tr_str = parameters.describe_repetition_time(metadata)
    te_str = parameters.describe_echo_times_fmap(files)
    fa_str = parameters.describe_flip_angle(metadata)
    fov_str, matrixsize_str, voxelsize_str = parameters.describe_image_size(img)
    mb_str = parameters.describe_multiband_factor(metadata)

    parameters_str = [
        dir_str,
        slice_str,
        tr_str,
        te_str,
        fa_str,
        fov_str,
        matrixsize_str,
        voxelsize_str,
        mb_str,
    ]
    parameters_str = [d for d in parameters_str if len(d)]
    parameters_str = "; ".join(parameters_str)

    for_str = parameters.describe_intendedfor_targets(metadata, layout)

    desc = (
        "A {variants} {seqs} field map ({parameters_str}) was "
        "acquired{for_str}.".format(
            variants=variants,
            seqs=seqs,
            for_str=for_str,
            parameters_str=parameters_str,
        )
    )
    return desc


def general_acquisition_info(metadata):
    """General sentence on data acquisition.

    This should be the first sentence in the MRI data acquisition section.

    Parameters
    ----------
    metadata : :obj:`dict`
        The metadata for the dataset.

    Returns
    -------
    out_str : :obj:`str`
        Output string with scanner information.
    """
    out_str = (
        "MR data were acquired using a {tesla}-Tesla {manu} {model} MRI "
        "scanner.".format(
            tesla=metadata.get("MagneticFieldStrength", "UNKNOWN"),
            manu=metadata.get("Manufacturer", "MANUFACTURER"),
            model=metadata.get("ManufacturersModelName", "MODEL"),
        )
    )
    return out_str


def final_paragraph(metadata):
    """Describe dicom-to-nifti conversion process and methods generation.

    Parameters
    ----------
    metadata : :obj:`dict`
        The metadata for the scan.

    Returns
    -------
    desc : :obj:`str`
        Output string with scanner information.
    """
    if "ConversionSoftware" in metadata.keys():
        soft = metadata["ConversionSoftware"]
        vers = metadata["ConversionSoftwareVersion"]
        software_str = " using {soft} ({conv_vers})".format(soft=soft, conv_vers=vers)
    else:
        software_str = ""
    desc = (
        "Dicoms were converted to NIfTI-1 format{software_str}. "
        "This section was (in part) generated automatically using pybids "
        "({meth_vers}).".format(
            software_str=software_str,
            meth_vers=__version__,
        )
    )
    return desc


def parse_files(layout, data_files, sub, config):
    """Loop through files in a BIDSLayout and generate appropriate descriptions.

    Then, compile all of the descriptions into a list.

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
    # Group files into individual runs
    data_files = collect_associated_files(layout, data_files, extra_entities=["run"])

    # print(data_files)

    description_list = []
    # Assume all data have same basic info
    description_list.append(general_acquisition_info(data_files[0][0].get_metadata()))

    for group in data_files:

        if group[0].entities["datatype"] == "func":
            group_description = func_info(layout, group, config)

        elif (group[0].entities["datatype"] == "anat") and group[0].entities[
            "suffix"
        ].endswith("w"):
            group_description = anat_info(layout, group, config)

        elif group[0].entities["datatype"] == "dwi":
            group_description = dwi_info(layout, group, config)

        elif (group[0].entities["datatype"] == "fmap") and group[0].entities[
            "suffix"
        ] == "phasediff":
            group_description = fmap_info(layout, group, config)

        elif group[0].entities["datatype"] in ["eeg", "meg", "beh", "perf"]:
            warnings.warn(group[0].entities["datatype"] + " not yet supported.")
            continue

        else:
            warnings.warn(group[0].filename + " not yet supported.")
            continue

        description_list.append(group_description)

    return description_list
