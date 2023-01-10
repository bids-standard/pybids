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


def nb_runs_str(nb_runs: int):
    if nb_runs == 1:
        return f"{num2words(nb_runs).title()} run"
    else:
        return f"{num2words(nb_runs).title()} runs"


""


def template_mri_info(desc_data):
    return f"""repetition time, TR={desc_data["tr"]}ms; echo time, TE={desc_data["te"]}ms; 
    flip angle, FA={desc_data["fa"]}<deg>; 
field of view, FOV={desc_data["fov"]}mm; 
matrix size={desc_data["matrix_size"]}; 
voxel size={desc_data["voxel_size"]}mm;"""


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
    task_name = metadata.get("TaskName", task_name)

    seqs, variants = parameters.describe_sequence(metadata, config)

    all_runs = sorted(list({f.get_entities().get("run", 1) for f in files}))

    desc_data = {
        "tr": parameters.repetition_time_ms(metadata),
        "fa": metadata.get("FlipAngle", "UNKNOWN"),
        "fov": parameters.field_of_view(img),
        "matrix_size": parameters.matrix_size(img),
        "voxel_size": parameters.voxel_size(img),
        "slice_timing": parameters.describe_slice_timing(img, metadata),
        "te": parameters.echo_time_ms(files),
        "mb_str": parameters.describe_multiband_factor(metadata),
        "inplaneaccel_str": parameters.describe_inplane_accel(metadata),
        "nb_runs": len(all_runs),
        "task_name": task_name,
        "variants": variants,
        "seqs": seqs,
        "multi_echo": parameters.multi_echo(files),
        "nb_vols": parameters.describe_nb_vols(all_imgs),
        "duration": parameters.describe_duration(all_imgs, metadata),
    }

    return template_func_info(desc_data)


def template_func_info(desc_data):
    return f"""{nb_runs_str(desc_data["nb_runs"])} of {desc_data["task_name"]} {desc_data["variants"]} 
{desc_data["seqs"]} {desc_data["multi_echo"]} fMRI data were collected 
({template_mri_info(desc_data)}  {desc_data["slice_timing"]}; 
{desc_data["te"]}; {desc_data["mb_str"]}; {desc_data["inplaneaccel_str"]}). 
Run duration was {desc_data["duration"]} minutes, during which {desc_data["nb_vols"]} volumes were acquired."""


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

    # General info
    seqs, variants = parameters.describe_sequence(metadata, config)

    all_runs = sorted(list({f.get_entities().get("run", 1) for f in files}))

    desc_data = {
        "tr": parameters.repetition_time_ms(metadata),
        "te": parameters.echo_time_ms(files),
        "fa": metadata.get("FlipAngle", "UNKNOWN"),
        "fov": parameters.field_of_view(img),
        "matrix_size": parameters.matrix_size(img),
        "voxel_size": parameters.voxel_size(img),
        "scan_type": first_file.get_entities()["suffix"].replace("w", "-weighted"),
        "slice_timing": parameters.describe_slice_timing(img, metadata),
        "nb_runs": len(all_runs),
        "variants": variants,
        "seqs": seqs,
        "multi_echo": parameters.multi_echo(files),
    }

    return template_anat_info(desc_data)


def template_anat_info(desc_data):
    return f"""{nb_runs_str(desc_data["nb_runs"])} of {desc_data["scan_type"]} {desc_data["variants"]} 
{desc_data["seqs"]} {desc_data["multi_echo"]} structural MRI data were collected 
({template_mri_info(desc_data)} {desc_data["slice_timing"]}; {desc_data["te"]})."""


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

    # General info
    seqs, variants = parameters.describe_sequence(metadata, config)

    all_runs = sorted(list({f.get_entities().get("run", 1) for f in files}))

    desc_data = {
        "tr": parameters.repetition_time_ms(metadata),
        "te": parameters.echo_time_ms(files),
        "fa": metadata.get("FlipAngle", "UNKNOWN"),
        "fov": parameters.field_of_view(img),
        "matrix_size": parameters.matrix_size(img),
        "voxel_size": parameters.voxel_size(img),
        "nb_runs": len(all_runs),
        "variants": variants,
        "seqs": seqs,
        "bval_str": parameters.describe_bvals(bval_file),
        "nvec_str": parameters.describe_dmri_directions(img),
        "mb_str": parameters.describe_multiband_factor(metadata),
    }

    return template_dwi_info(desc_data)


def template_dwi_info(desc_data):
    return f"""{nb_runs_str(desc_data["nb_runs"])} of {desc_data["variants"]} 
{desc_data["seqs"]} diffusion-weighted (dMRI) data were collected ({template_mri_info(desc_data)} 
{desc_data["bval_str"]}; {desc_data["nvec_str"]}; {desc_data["mb_str"]};
{desc_data["te"]};)."""


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
    desc_data = {
        "tr": parameters.repetition_time_ms(metadata),
        "te": parameters.describe_echo_times_fmap(files),
        "fa": metadata.get("FlipAngle", "UNKNOWN"),
        "fov": parameters.field_of_view(img),
        "matrix_size": parameters.matrix_size(img),
        "voxel_size": parameters.voxel_size(img),
        "slice_timing": parameters.describe_slice_timing(img, metadata),
        "dir_str": parameters.describe_pe_direction(metadata, config),
        "variants": variants,
        "seqs": seqs,
        "mb_str": parameters.describe_multiband_factor(metadata),
        "for_str": parameters.describe_intendedfor_targets(metadata, layout),
    }

    return template_fmap_info(desc_data)


def template_fmap_info(desc_data):
    return f"""A {desc_data["variants"]} {desc_data["seqs"]} fieldmap ({template_mri_info(desc_data)}
{desc_data["mb_str"]}; {desc_data["slice_timing"]}; {desc_data["te"]};) was acquired {desc_data["for_str"]}.
"""


def template_general_acquisition_info(desc_data):
    return f"""MR data were acquired using a {desc_data["tesla"]}-Tesla 
    {desc_data["manufacturer"]} {desc_data["model"]}."""


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

    return template_general_acquisition_info(desc_data)


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


def footer():
    return (
        "This section was (in part) generated automatically using pybids {__version__}."
    )


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
