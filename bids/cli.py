import os
import json
from copy import deepcopy
from pathlib import Path
import click

from . import __version__
from .layout import BIDSLayoutIndexer, BIDSLayout
from .utils import validate_multiple as _validate_multiple
from .modeling.report.base import generate_report

# alias -h to trigger help message
CONTEXT_SETTINGS = {'help_option_names': ['-h', '--help']}


class Either(click.ParamType):
    """Click type that will accept any of the types passed at initialization.

    Examples
    --------
    >>> Either(click.BOOL, click.STRING).convert(True, None, None)
    True
    >>> Either(click.BOOL, click.STRING).convert("Test", None, None)
    'Test'

    Note that the order of types can affect the interpreted type.

    >>> Either(click.BOOL, click.INT).convert("1", None, None)
    True
    >>> Either(click.INT, click.BOOL).convert("1", None, None)
    1

    The motivating use case is True or path:

    >>> PathOrTrue = Either(click.BOOL, click.Path(exists=True))
    >>> PathOrTrue.convert(True, None, None)
    True
    >>> cwd = os.getcwd()
    >>> PathOrTrue.convert(cwd, None, None) == cwd
    True
    >>> PathOrTrue.convert('/does/not/exist', None, None)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    BadParameter: '/does/not/exist' does not match expected types:
    boolean: "'/does/not/exist' is not a valid boolean."
    path: "'/does/not/exist' does not exist."
    """
    def __init__(self, *types):
        self.types = types

    @property
    def name(self):
        tpstrings = tuple(tp.name for tp in self.types)
        return "any type in {tpstrings}"

    def convert(self, value, param, ctx):
        errors = []
        for tp in self.types:
            try:
                return tp.convert(value, param, ctx)
            except click.BadParameter as err:
                errors.append(f"{tp.name}: {err.message!r}")
        messages = "\n".join(errors)
        self.fail(f"{value!r} does not match expected types:\n{messages}")


class PathOrRegex(click.ParamType):
    "A helper Type to parse BIDSLayoutIndexer ignore/force entries"
    name = "path or m/regex/"

    def convert(self, value, param, ctx):
        import re
        if re.match(r"^m/.*/$", value):  # has form "m/<regex>/"
            value = re.compile(value[2:-1])
        return value


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(__version__, prog_name='pybids')
def cli():
    """Command-line interface for PyBIDS operations"""
    pass


@cli.command(context_settings=CONTEXT_SETTINGS)
@click.argument('root', type=click.Path(file_okay=False, exists=True))
@click.argument('db-path', type=click.Path(file_okay=False, resolve_path=True, exists=True))
@click.option('--derivatives', multiple=True, default=[False], is_flag=False, flag_value=True,
              show_default=True, type=Either(click.BOOL, click.Path(exists=True)),
              help="Specifies whether and/or which derivatives to index.")
@click.option('--reset-db', default=False, show_default=True, is_flag=True,
              help="Remove existing database index if present.")
@click.option('--validate/--no-validate', default=True, show_default=True,
              help="Check for BIDS compliance when indexing files.")
@click.option('--config', multiple=True,
              help="Optional name(s) of configuration file(s) to use.")
@click.option('--index-metadata/--no-index-metadata', default=False, show_default=True,
              help="Include metadata when indexing files.")
@click.option('--ignore', multiple=True, type=PathOrRegex(),
              help="Path (from root) or regex to exclude from indexing. "
                   "Regex entries need to fitted with leading 'm/' and trailing '/'.")
@click.option('--force-index', multiple=True, type=PathOrRegex(),
              help="Path (from root) or regex to include when indexing. "
                   "Regex entries need to fitted with leading 'm/' and trailing '/'.")
@click.option('--config-filename', type=click.Path(),
              default="layout_config.json", show_default=True,
              help="Name of filename within directories that contains configuration information.")
def layout(
    root,
    db_path,
    derivatives,
    reset_db,
    validate,
    config,
    index_metadata,
    ignore,
    force_index,
    config_filename,
):
    """
    Initialize a BIDSLayout, and create an SQLite database index.
    """

    # ensure empty multiples are set to None
    derivatives = _validate_multiple(derivatives, retval=False)
    config = _validate_multiple(config)
    ignore = _validate_multiple(ignore)
    force_index = _validate_multiple(force_index)

    if not (Path(db_path) / 'layout_index.sqlite').exists():
        reset_db = True

    layout = BIDSLayout(
        root,
        database_path=db_path,
        reset_database=reset_db,
        validate=validate,
        config=config,
        indexer=BIDSLayoutIndexer(
            validate=validate,
            index_metadata=index_metadata,
            ignore=ignore,
            force_index=force_index,
            config_filename=config_filename,
        ),
    )
    if reset_db:
        click.echo("Successfully generated database index at {}".format(db_path))
    else:
        click.echo(
            "Previously generated database index found at {}. "
            "To generate a new index, rerun with ``--reset-db``".format(db_path)
        )


@cli.command(context_settings=CONTEXT_SETTINGS)
@click.argument('root', type=click.Path(file_okay=False, exists=True))
def upgrade(root):
    """
    Upgrade common experimental BIDS features to finalized versions.
    """

    click.echo(
        "WARNING. This upgrade tool is EXPERIMENTAL and MAY damage your "
        "dataset. Please ensure you have a backup before proceeding."
    )
    click.confirm("Proceed?", abort=True)
    changes = False

    description_path = Path(root) / "dataset_description.json"
    orig_desc = json.loads(description_path.read_text())
    desc = upgrade_dataset_description(orig_desc)

    if desc != orig_desc:
        description_path.write_text(json.dumps(desc))
        changes = True

    val = click.prompt("Load dataset and update filenames?", default="Y",
                       type=click.Choice("YN"))
    if val == "Y":
        upgrade_filenames(root, desc)

    if changes:
        click.echo("Upgrade complete. Please run the bids-validator "
                   "(https://bids-standard.github.io/bids-validator) "
                   "to confirm the correctness of the changes.")
    else:
        click.echo("No changes to make!")


def upgrade_dataset_description(description):
    """
    Upgrade dataset_description.json with recommended values
    """
    description = deepcopy(description)

    # Give an opportunity to update to latest version
    bidsver = description.get("BIDSVersion")
    if bidsver is None or bidsver < "1.6.0":
        val = click.prompt(f"Update BIDS Version? (current: {bidsver})",
                           default="1.6.0", type=str)
        if val.startswith("1."):
            description["BIDSVersion"] = val
        else:
            click.echo("Expected version to be 1.x, e.g., 1.6.0. Skipping.")

    # Always update DatasetType if missing
    if "DatasetType" not in description:
        val = click.prompt("Is this dataset [r]aw or [d]erivative?", default="r",
                           type=click.Choice(("r", "d"), case_sensitive=False))
        description["DatasetType"] = "raw" if val == "R" else "derivative"

    if description["DatasetType"] == "derivative":
        if "PipelineDescription" in description:
            val = click.prompt("Convert PipelineDescription to GeneratedBy?", default="Y",
                               type=click.Choice("YN"))
            if val == "Y":
                description["GeneratedBy"] = [description.pop("PipelineDescription")]

    return description


def upgrade_filenames(root, description):
    dstype = description["DatasetType"]
    layout = BIDSLayout(root, validate=False,
                        config="bids" if dstype == "raw" else "derivatives")

    # Rename regressors.tsv to timeseries.tsv
    regressors = layout.get(suffix="regressors")
    policy = None
    with click.progressbar(regressors) as bar:
        for bidsfile in bar:
            action = policy
            new_path = bidsfile.path.replace("regressors.", "timeseries.")
            if action is None:
                action = click.prompt(
                    f"Rename {bidsfile.path} to {new_path}? ([y]es/[n]o/[A]ll/[N]one)", default="y",
                    type=click.Choice("ynAN"), show_choices=False)
                if action in "AN":
                    policy = action
            if action in "yA":
                click.echo(f"Renaming {bidsfile.path} -> {new_path}")
                os.rename(bidsfile.path, new_path)
            else:
                click.echo(f"Not renaming {bidsfile.path}")


@cli.command(context_settings=CONTEXT_SETTINGS)
@click.argument('model', type=click.Path(file_okay=True, exists=True, dir_okay=False))
@click.argument('root', type=click.Path(file_okay=False, exists=True))
@click.argument('derivatives', type=click.Path(file_okay=False, exists=True), nargs=-1)
@click.argument('output_dir', type=click.Path(file_okay=False, exists=False))
@click.option('--scan_length', type=click.FLOAT)
@click.option('--sub_label', '-s', multiple=True, 
    help='The label(s) of the subjects(s) to create report for')
@click.option('--ses_label', '-ss', multiple=True, 
    help='The label(s) of the session(s) to create report for')
@click.option('--task_label', '-t', multiple=True,
    help='The label(s) of the task(s) to create report for')
@click.option('--run_label', '-r', multiple=True,
    help='The label(s) of the run(s) to create report for')
def model_report(model, root, derivatives, output_dir, scan_length, 
    sub_label, ses_label, task_label, run_label):
    """
    Generate a report of a BIDS StatsModel.
    """
    entities = {'subject': sub_label, 'session': ses_label,
                'task': task_label, 'run': run_label}

    entities = {key:val for key, val in entities.items() if val}
    generate_report(
            model, root, derivatives, output_dir, scan_length, **entities
    )