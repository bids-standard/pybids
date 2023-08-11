"""Functionality related to validation of BIDSLayouts and BIDS projects."""

from pathlib import Path
import json
import re
import warnings

from ..utils import listify
from ..exceptions import BIDSValidationError, BIDSDerivativesValidationError


MANDATORY_BIDS_FIELDS = {
    "Name": {"Name": "Example dataset"},
    "BIDSVersion": {"BIDSVersion": "1.0.2"},
}


MANDATORY_DERIVATIVES_FIELDS = {
    **MANDATORY_BIDS_FIELDS,
    "GeneratedBy": {
        "GeneratedBy": [{"Name": "Example pipeline"}]
    },
}

EXAMPLE_BIDS_DESCRIPTION = {
    k: val[k] for val in MANDATORY_BIDS_FIELDS.values() for k in val}


EXAMPLE_DERIVATIVES_DESCRIPTION = {
    k: val[k] for val in MANDATORY_DERIVATIVES_FIELDS.values() for k in val}


DEFAULT_LOCATIONS_TO_IGNORE = {
    re.compile(r"^/(code|models|sourcedata|stimuli)"),
    re.compile(r'/\.'),
}

def absolute_path_deprecation_warning():
    warnings.warn("The absolute_paths argument will be removed from PyBIDS "
                  "in 0.14. You can easily access the relative path of "
                  "BIDSFile objects via the .relpath attribute (instead of "
                  ".path). Switching to this pattern is strongly encouraged, "
                  "as the current implementation of relative path handling "
                  "is known to produce query failures in certain edge cases.")


def indexer_arg_deprecation_warning():
    warnings.warn("The ability to pass arguments to BIDSLayout that control "
                  "indexing is likely to be removed in future; possibly as "
                  "early as PyBIDS 0.14. This includes the `config_filename`, "
                  "`ignore`, `force_index`, and `index_metadata` arguments. "
                  "The recommended usage pattern is to initialize a new "
                  "BIDSLayoutIndexer with these arguments, and pass it to "
                  "the BIDSLayout via the `indexer` argument.")


def validate_root(root, validate):
    # Validate root argument and make sure it contains mandatory info
    try:
        root = Path(root)
    except TypeError:
        raise TypeError("root argument must be a pathlib.Path (or a type that "
                        "supports casting to pathlib.Path, such as "
                        "string) specifying the directory "
                        "containing the BIDS dataset.")

    root = root.absolute()

    if not root.exists():
        raise ValueError("BIDS root does not exist: %s" % root)

    target = root / 'dataset_description.json'
    if not target.exists():
        if validate:
            raise BIDSValidationError(
                "'dataset_description.json' is missing from project root."
                " Every valid BIDS dataset must have this file."
                "\nExample contents of 'dataset_description.json': \n%s" %
                json.dumps(EXAMPLE_BIDS_DESCRIPTION)
            )
        else:
            description = None
    else:
        err = None
        try:
            with open(target, 'r', encoding='utf-8') as desc_fd:
                description = json.load(desc_fd)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            description = None
            err = e
        if validate:

            if description is None:
                raise BIDSValidationError(
                    "'dataset_description.json' is not a valid json file."
                    " There is likely a typo in your 'dataset_description.json'."
                    "\nExample contents of 'dataset_description.json': \n%s" %
                    json.dumps(EXAMPLE_BIDS_DESCRIPTION)
                ) from err

            for k in MANDATORY_BIDS_FIELDS:
                if k not in description:
                    raise BIDSValidationError(
                        "Mandatory %r field missing from "
                        "'dataset_description.json'."
                        "\nExample: %s" % (k, MANDATORY_BIDS_FIELDS[k])
                    )

    return root, description


def validate_derivative_path(path, **kwargs):
    # Collect all paths that contain a dataset_description.json
    dd = Path(path) / 'dataset_description.json'
    description = json.loads(dd.read_text(encoding='utf-8'))
    pipeline_names = [pipeline["Name"]
                      for pipeline in description.get("GeneratedBy", [])
                      if "Name" in pipeline]
    if pipeline_names:
        pipeline_name = pipeline_names[0]
    elif "PipelineDescription" in description:
        warnings.warn("The PipelineDescription field was superseded "
                      "by GeneratedBy in BIDS 1.4.0. You can use "
                      "``pybids upgrade`` to update your derivative "
                      "dataset.")
        pipeline_name = description["PipelineDescription"].get("Name")
    else:
        pipeline_name = None
    if pipeline_name is None:
        raise BIDSDerivativesValidationError(
            "Every valid BIDS-derivatives dataset must "
            "have a GeneratedBy.Name field set "
            "inside 'dataset_description.json'. "
            f"\nExample: {MANDATORY_DERIVATIVES_FIELDS['GeneratedBy']}"
        )
    return pipeline_name


def _sort_patterns(patterns, root):
    """Return sorted patterns, from more specific to more general."""
    regexes = [patt for patt in patterns if hasattr(patt, "search")]

    paths = [
        str((root / patt).absolute())
        for patt in listify(patterns)
        if not hasattr(patt, "search")
    ]
    # Sort patterns from general to specific
    paths.sort(key=len)

    # Combine and return (note path patterns are reversed, specific first)
    return [Path(p) for p in reversed(paths)] + regexes


def validate_indexing_args(ignore, force_index, root):
    if ignore is None:
        ignore = list(
            DEFAULT_LOCATIONS_TO_IGNORE - set(force_index or [])
        )

    # root has already been validated to be a directory
    ignore = _sort_patterns(ignore, root)
    force_index = _sort_patterns(force_index or [], root)

    # Derivatives get special handling; they shouldn't be indexed normally
    for entry in force_index:
        condi = (isinstance(entry, str) and
                 str(entry.resolve()).startswith('derivatives'))
        if condi:
            msg = ("Do not pass 'derivatives' in the force_index "
                    "list. To index derivatives, either set "
                    "derivatives=True, or use add_derivatives().")
            raise ValueError(msg)

    return ignore, force_index
