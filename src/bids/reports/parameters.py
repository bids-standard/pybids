"""Functions for building strings for individual parameters."""
import logging
import math
import os
import os.path as op

import nibabel as nib
from num2words import num2words

from .utils import list_to_str, num_to_str, remove_duplicates

logging.basicConfig()
LOGGER = logging.getLogger("pybids.reports.parsing")


def describe_slice_timing(img, metadata: dict) -> str:
    """Generate description of slice timing from metadata."""

    if "SliceTiming" in metadata:
        slice_order = " in {0} order".format(get_slice_info(metadata["SliceTiming"]))
        n_slices = len(metadata["SliceTiming"])
    else:
        slice_order = ""
        n_slices = img.shape[2]

    return "{n_slices} slices{slice_order}".format(
        n_slices=n_slices, slice_order=slice_order
    )


def describe_repetition_time(metadata: dict):
    """Generate description of repetition time from metadata."""
    tr = metadata["RepetitionTime"] * 1000
    tr = num_to_str(tr)
    return "repetition time, TR={tr}ms".format(tr=tr)


def describe_func_duration(n_vols: int, tr) -> str:
    """Generate description of functional run length from repetition time and number of volumes."""
    run_secs = math.ceil(n_vols * tr)
    mins, secs = divmod(run_secs, 60)
    return "{0}:{1:02.0f}".format(int(mins), int(secs))


def describe_duration(files) -> str:
    """Generate general description of scan length from files."""
    first_file = files[0]
    metadata = first_file.get_metadata()

    tr = metadata["RepetitionTime"]
    imgs = [nib.load(f) for f in files]
    n_vols = [img.shape[3] for img in imgs]

    if len(set(n_vols)) > 1:
        min_vols = min(n_vols)
        max_vols = max(n_vols)
        min_dur = describe_func_duration(min_vols, tr)
        max_dur = describe_func_duration(max_vols, tr)
        dur_str = "{}-{}".format(min_dur, max_dur)
        n_vols = "{}-{}".format(min_vols, max_vols)

    else:
        n_vols = n_vols[0]
        dur_str = describe_func_duration(n_vols, tr)

    dur_str = (
        "Run duration was {0} minutes, during which {1} volumes were acquired."
    ).format(dur_str, n_vols)
    return dur_str


def describe_multiband_factor(metadata) -> str:
    """Generate description of the multi-band acceleration applied, if used."""
    return (
        "MB factor={}".format(metadata["MultibandAccelerationFactor"])
        if metadata.get("MultibandAccelerationFactor", 1) > 1
        else ""
    )


def describe_echo_times(files):
    """Generate description of echo times from metadata field.

    Parameters
    ----------
    files : :obj:`list` of :obj:`bids.layout.models.BIDSFile`
        List of nifti files in layout corresponding to file collection.

    Returns
    -------
    te_str : str
        Description of echo times.
    me_str : str
        Whether the data are multi-echo or single-echo.
    """

    echo_times = [f.get_metadata()["EchoTime"] for f in files]
    echo_times = sorted(list(set(echo_times)))
    if len(echo_times) > 1:
        te = [num_to_str(t * 1000) for t in echo_times]
        te = list_to_str(te)
        me_str = "multi-echo"
    else:
        te = num_to_str(echo_times[0] * 1000)
        me_str = "single-echo"
    te_str = "echo time, TE={}ms".format(te)
    return te_str, me_str


def describe_echo_times_fmap(files):
    """Generate description of echo times from metadata field for fmaps

    Parameters
    ----------
    files : :obj:`list` of :obj:`bids.layout.models.BIDSFile`
        List of nifti files in layout corresponding to file collection.

    Returns
    -------
    te_str : str
        Description of echo times.
    """
    # TODO handle all types of fieldmaps

    echo_times1 = [f.get_metadata()["EchoTime1"] for f in files]
    echo_times2 = [f.get_metadata()["EchoTime2"] for f in files]
    echo_times1 = sorted(list(set(echo_times1)))
    echo_times2 = sorted(list(set(echo_times2)))
    if len(echo_times1) <= 1 and len(echo_times2) <= 1:
        # if that's not the case we should probably throw a warning
        # because we should expect the same echo times for all values
        te1 = num_to_str(echo_times1[0] * 1000)
        te2 = num_to_str(echo_times2[0] * 1000)
    return "echo time 1 / 2, TE1/2={0}{1}ms".format(te1, te2)


def describe_image_size(img):
    """Generate description imaging data sizes, including FOV, voxel size, and matrix size.

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
    vs_str, ms_str, fov_str = get_size_str(img)

    fov_str = "field of view, FOV={}mm".format(fov_str)
    voxelsize_str = "voxel size={}mm".format(vs_str)
    matrixsize_str = "matrix size={}".format(ms_str)

    return fov_str, matrixsize_str, voxelsize_str


def describe_inplane_accel(metadata: dict) -> str:
    """Generate description of in-plane acceleration factor, if any."""
    return (
        "in-plane acceleration factor={}".format(
            metadata["ParallelReductionFactorInPlane"]
        )
        if metadata.get("ParallelReductionFactorInPlane", 1) > 1
        else ""
    )


def describe_flip_angle(metadata: dict) -> str:
    """Generate description of flip angle."""
    return "flip angle, FA={}<deg>".format(metadata.get("FlipAngle", "UNKNOWN"))


def describe_dmri_directions(img):
    """Generate description of diffusion directions."""
    return "{} diffusion directions".format(img.shape[3])


def describe_bvals(bval_file) -> str:
    """Generate description of dMRI b-values."""
    # Parse bval file
    with open(bval_file, "r") as file_object:
        raw_bvals = file_object.read().splitlines()
    # Flatten list of space-separated values
    bvals = [
        item for sublist in [line.split(" ") for line in raw_bvals] for item in sublist
    ]
    bvals = sorted([int(v) for v in set(bvals)])
    bvals = [num_to_str(v) for v in bvals]
    bval_str = list_to_str(bvals)
    bval_str = "b-values of {} acquired".format(bval_str)
    return bval_str


def describe_pe_direction(metadata: dict, config: dict) -> str:
    """Generate description of phase encoding direction."""
    dir_str = config["dir"][metadata["PhaseEncodingDirection"]]
    dir_str = "phase encoding: {}".format(dir_str)
    return dir_str


def describe_intendedfor_targets(metadata: dict, layout) -> str:
    """Generate description of intended for targets."""
    if "IntendedFor" in metadata.keys():

        scans = metadata["IntendedFor"]
        run_dict = {}

        for scan in scans:
            fn = op.basename(scan)
            if_file = [
                f for f in layout.get(extension=[".nii", ".nii.gz"]) if fn in f.path
            ][0]
            run_num = int(if_file.run)
            target_type = if_file.entities["suffix"].upper()

            if target_type == "BOLD":
                iff_meta = layout.get_metadata(if_file.path)
                task = iff_meta.get("TaskName", if_file.entities["task"])
                target_type_str = "{0} {1} scan".format(task, target_type)
            else:
                target_type_str = "{0} scan".format(target_type)

            if target_type_str not in run_dict.keys():
                run_dict[target_type_str] = []

            run_dict[target_type_str].append(run_num)

        for scan in run_dict.keys():
            run_dict[scan] = [
                num2words(r, ordinal=True) for r in sorted(run_dict[scan])
            ]

        out_list = []

        for scan in run_dict.keys():

            if len(run_dict[scan]) > 1:
                s = "s"
            else:
                s = ""

            run_str = list_to_str(run_dict[scan])
            string = "{rs} run{s} of the {sc}".format(rs=run_str, s=s, sc=scan)
            out_list.append(string)

        for_str = " for the {0}".format(list_to_str(out_list))

    else:
        for_str = ""

    return for_str


def get_slice_info(slice_times) -> str:
    """Extract slice order from slice timing info.

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

    if slice_order == list(range(len(slice_order))):
        slice_order_name = "sequential ascending"

    elif slice_order == list(reversed(range(len(slice_order)))):
        slice_order_name = "sequential descending"

    elif slice_order[0] < slice_order[1]:
        # We're allowing some wiggle room on interleaved.
        slice_order_name = "interleaved ascending"

    elif slice_order[0] > slice_order[1]:
        slice_order_name = "interleaved descending"

    else:
        slice_order = [str(s) for s in slice_order]
        raise Exception("Unknown slice order: [{0}]".format(", ".join(slice_order)))

    return slice_order_name


def describe_sequence(metadata: dict, config: dict):
    """Extract and reformat imaging sequence(s) and variant(s) into pretty strings.

    Parameters
    ----------
    metadata : :obj:`dict`
        The metadata for the scan.
    config : :obj:`dict`
        A dictionary with relevant information regarding sequences, sequence
        variants, phase encoding directions, and task names.

    Returns
    -------
    seqs : :obj:`str`
        Sequence names.
    variants : :obj:`str`
        Sequence variant names.
    """
    seq_abbrs = metadata.get("ScanningSequence", "").split("_")
    seqs = [config["seq"].get(seq, seq) for seq in seq_abbrs]
    seqs = list_to_str(seqs)
    if seq_abbrs[0]:
        seqs += " ({0})".format(os.path.sep.join(seq_abbrs))

    variants = [
        config["seqvar"].get(var, var)
        for var in metadata.get("SequenceVariant", "").split("_")
    ]
    variants = list_to_str(variants)

    return seqs, variants


def get_size_str(img):
    """Extract and reformat voxel size, matrix size, FOV, and number of slices into strings.

    Parameters
    ----------
    img : :obj:`nibabel.Nifti1Image`
        Image from scan from which to derive parameters.

    Returns
    -------
    voxel_size : :obj:`str`
        Voxel size string (e.g., '2x2x2')
    matrix_size : :obj:`str`
        Matrix size string (e.g., '128x128')
    fov : :obj:`str`
        Field of view string (e.g., '256x256')
    """
    n_x, n_y = img.shape[:2]
    import numpy as np

    matrix_size = "{0}x{1}".format(num_to_str(n_x), num_to_str(n_y))

    voxel_dims = np.array(img.header.get_zooms()[:3])

    voxel_size = "x".join([num_to_str(s) for s in voxel_dims])


    fov = [n_x, n_y] * voxel_dims[:2]
    fov = "x".join([num_to_str(s) for s in fov])

    return voxel_size, matrix_size, fov
