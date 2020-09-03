from pathlib import Path
import click

from . import __version__
from .layout import BIDSLayoutIndexer, BIDSLayout
from .utils import validate_multiple as _validate_multiple

# alias -h to trigger help message
CONTEXT_SETTINGS = {'help_option_names': ['-h', '--help']}


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
@click.option('--derivatives', multiple=True, default=False, show_default=True, flag_value=True,
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
