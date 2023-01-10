"""Parsing functions for generating BIDSReports."""
import logging
import warnings

import nibabel as nib

from ..utils import collect_associated_files
from . import parameters, templates

logging.basicConfig()
LOGGER = logging.getLogger("pybids.reports.parsing")


def func_info(files, config):
    """Generate a paragraph describing T2*-weighted functional scans.

    Parameters
    ----------
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
    all_imgs = [nib.load(f) for f in files]

    # General info
    task_name = first_file.get_entities()["task"] + " task"

    all_runs = sorted(list({f.get_entities().get("run", 1) for f in files}))

    desc_data = {
        "tr": metadata["RepetitionTime"] * 1000,
        "te": parameters.echo_time_ms(files),
        "fa": metadata.get("FlipAngle", "UNKNOWN"),
        "fov": parameters.field_of_view(img),
        "matrix_size": parameters.matrix_size(img),
        "voxel_size": parameters.voxel_size(img),
        "nb_slices": parameters.nb_slices(img, metadata),
        "slice_order": parameters.slice_order(metadata),
        "mb_str": parameters.multiband_factor(metadata),
        "inplaneaccel_str": parameters.inplane_accel(metadata),
        "nb_runs": len(all_runs),
        "task_name": metadata.get("TaskName", task_name),
        "variants": parameters.variants(metadata, config),
        "seqs": parameters.sequence(metadata, config),
        "multi_echo": parameters.multi_echo(files),
        "nb_vols": parameters.nb_vols(all_imgs),
        "duration": parameters.duration(all_imgs, metadata),
    }

    return templates.func_info(desc_data)


def anat_info(files, config):
    """Generate a paragraph describing T1- and T2-weighted structural scans.

    Parameters
    ----------
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

    all_runs = sorted(list({f.get_entities().get("run", 1) for f in files}))

    desc_data = {
        "tr": metadata["RepetitionTime"] * 1000,
        "te": parameters.echo_time_ms(files),
        "fa": metadata.get("FlipAngle", "UNKNOWN"),
        "fov": parameters.field_of_view(img),
        "matrix_size": parameters.matrix_size(img),
        "voxel_size": parameters.voxel_size(img),
        "nb_slices": parameters.nb_slices(img, metadata),
        "slice_order": parameters.slice_order(metadata),
        "scan_type": first_file.get_entities()["suffix"].replace("w", "-weighted"),
        "nb_runs": len(all_runs),
        "variants": parameters.variants(metadata, config),
        "seqs": parameters.sequence(metadata, config),
        "multi_echo": parameters.multi_echo(files),
    }

    return templates.anat_info(desc_data)


def dwi_info(files, config):
    """Generate a paragraph describing DWI scan acquisition information.

    Parameters
    ----------
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

    all_runs = sorted(list({f.get_entities().get("run", 1) for f in files}))

    desc_data = {
        "tr": metadata["RepetitionTime"] * 1000,
        "te": parameters.echo_time_ms(files),
        "fa": metadata.get("FlipAngle", "UNKNOWN"),
        "fov": parameters.field_of_view(img),
        "matrix_size": parameters.matrix_size(img),
        "voxel_size": parameters.voxel_size(img),
        "nb_runs": len(all_runs),
        "variants": parameters.variants(metadata, config),
        "seqs": parameters.sequence(metadata, config),
        "bvals": parameters.bvals(bval_file),
        "dmri_dir": img.shape[3],
        "mb_str": parameters.multiband_factor(metadata),
    }

    return templates.dwi_info(desc_data)


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

    # Parameters
    desc_data = {
        "tr": metadata["RepetitionTime"] * 1000,
        "te": parameters.echo_times_fmap(files),
        "fa": metadata.get("FlipAngle", "UNKNOWN"),
        "fov": parameters.field_of_view(img),
        "matrix_size": parameters.matrix_size(img),
        "voxel_size": parameters.voxel_size(img),
        "nb_slices": parameters.nb_slices(img, metadata),
        "slice_order": parameters.slice_order(metadata),
        "dir": config["dir"][metadata["PhaseEncodingDirection"]],
        "variants": parameters.variants(metadata, config),
        "seqs": parameters.sequence(metadata, config),
        "mb_str": parameters.multiband_factor(metadata),
        "for_str": parameters.intendedfor_targets(metadata, layout),
    }

    return templates.fmap_info(desc_data)


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

    desc_data = {
        "tesla": metadata.get("MagneticFieldStrength", "UNKNOWN"),
        "manufacturer": metadata.get("Manufacturer", "MANUFACTURER"),
        "model": metadata.get("ManufacturersModelName", "MODEL"),
    }

    return templates.general_acquisition_info(desc_data)


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
    return f"Dicoms were converted to NIfTI-1 format{software_str}."


def parse_files(layout, data_files, config):
    """Loop through files in a BIDSLayout and generate appropriate descriptions.

    Then, compile all of the descriptions into a list.

    Parameters
    ----------
    layout : :obj:`bids.layout.BIDSLayout`
        Layout object for a BIDS dataset.
    data_files : :obj:`list` of :obj:`bids.layout.models.BIDSFile`
        List of nifti files in layout corresponding to subject/session combo.
    config : :obj:`dict`
        Configuration info for methods generation.
    """
    # Group files into individual runs
    data_files = collect_associated_files(layout, data_files, extra_entities=["run"])

    # print(data_files)

    description_list = [general_acquisition_info(data_files[0][0].get_metadata())]
    for group in data_files:

        if group[0].entities["datatype"] == "func":
            group_description = func_info(group, config)

        elif (group[0].entities["datatype"] == "anat") and group[0].entities[
            "suffix"
        ].endswith("w"):
            group_description = anat_info(group, config)

        elif group[0].entities["datatype"] == "dwi":
            group_description = dwi_info(group, config)

        elif (group[0].entities["datatype"] == "fmap") and group[0].entities[
            "suffix"
        ] == "phasediff":
            group_description = fmap_info(layout, group, config)

        elif group[0].entities["datatype"] in [
            "eeg",
            "meg",
            "pet",
            "ieeg",
            "beh",
            "perf",
            "fnirs",
            "microscopy",
        ]:
            warnings.warn(group[0].entities["datatype"] + " not yet supported.")
            continue

        else:
            warnings.warn(f"{group[0].filename} not yet supported.")
            continue

        description_list.append(group_description)

    return description_list
