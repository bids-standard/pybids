from pathlib import Path
import click

from . import __version__

# alias -h to trigger help message
CONTEXT_SETTINGS = {'help_option_names': ['-h', '--help']}


class PathOrRegex(click.ParamType):
    "A helper Type to parse BIDSLayoutIndexer ignore/force entries"
    name = "path or /regex/"

    def convert(self, value, param, ctx):
        if value.startswith('/') and value.endswith('/'):
            # regex pattern
            import re
            value = re.compile(value[1:-1])
        # otherwise, return as is
        return value


# create group of commands as entrypoint
@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(__version__, prog_name='pybids')
def cli():
    """Command-line interface for PyBIDS operations"""
    pass


@cli.command(context_settings=CONTEXT_SETTINGS)
@click.argument('root', type=click.Path(file_okay=False, exists=True))
@click.option('--db-path', type=click.Path(file_okay=False, resolve_path=True),
              help="Path to save database index.")
@click.option('--derivatives', multiple=True,
              help="Specifies whether and/or which derivatives to to index.")
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
                   "Regex entries need to fitted with leading and trailing '/'.")
@click.option('--force-index', multiple=True, type=PathOrRegex(),
              help="Path (from root) or regex to include when indexing. "
                   "Regex entries need to fitted with leading and trailing '/'.")
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
    Initialize a BIDSLayout.

    If ``--db-path`` is provided, an SQLite database index for this BIDS dataset will be created.

    """
    if db_path and not (Path(db_path) / 'layout_index.sqlite').exists():
        reset_db = True

    indexer = _init_indexer(
        validate=validate,
        index_metadata=index_metadata,
        ignore=ignore,
        force_index=force_index,
        config_filename=config_filename,
    )
    _init_layout(
        root,
        validate=validate,
        config=config,
        indexer=indexer,
    )
    if db_path and reset_db:
        click.echo("Successfully generated database index at {}".format(db_path))


def _init_indexer(**kwargs):
    from .layout import BIDSLayoutIndexer

    return BIDSLayoutIndexer(**kwargs)


def _init_layout(root, **kwargs):
    from .layout import BIDSLayout

    return BIDSLayout(root, **kwargs)
