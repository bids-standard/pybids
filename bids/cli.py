from pathlib import Path
import click

from . import __version__

# alias -h to trigger help message
CONTEXT_SETTINGS = {'help_option_names': ['-h', '--help']}


class BIDSIndexIgnoreRe(click.ParamType):
    "A helper Type to parse BIDSLayoutIndexer ignore entries"
    name = "ignore"

    def convert(self, value, param, ctx):
        import re
        # will this ever fail?
        return re.compile(value)


# create group of commands as entrypoint
@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(__version__, prog_name='pybids')
def cli():
    """Command-line interface for PyBIDS operations"""
    pass


@cli.command(context_settings=CONTEXT_SETTINGS)
@click.argument('root', type=click.Path(file_okay=False, exists=True))
@click.option('--output', type=click.Path(file_okay=False, resolve_path=True),
              default=Path('.').resolve(), help="Path to save database index")
@click.option('--skip-validation', default=False, is_flag=True,
              help="Skip check for BIDS compliance when indexing files.")
@click.option('--skip-metadata', default=False, is_flag=True,
              help="Skip metadata when indexing files.")
@click.option('--ignore-path', multiple=True, default=None,
              help="Path (from root) to exclude from indexing.")
@click.option('--ignore-regex', multiple=True, default=None, type=BIDSIndexIgnoreRe(),
              help="Regex to exclude from indexing.")
def save_db(root, output, skip_validation, skip_metadata, ignore_path, ignore_regex):
    """Initialize and save an SQLite database index for this BIDS dataset.

    If ``output`` contains a previously generated index, this method will raise a
    ``RuntimeError``.
    """
    if (Path(output) / 'layout_index.sqlite').exists():
        raise RuntimeError("Previous index exists at {}".format(output))

    ignore = None
    if ignore_path or ignore_regex:
        ignore = ignore_path + ignore_regex

    indexer = _init_indexer(
        validate=not skip_validation,
        index_metadata=not skip_metadata,
        ignore=ignore,
    )

    layout = _init_layout(
        root,
        validate=not skip_validation,
        indexer=indexer,
    )
    layout.save(output)
    click.echo("Successfully generated database index at {}".format(output))


def _init_indexer(**kwargs):
    from .layout import BIDSLayoutIndexer

    return BIDSLayoutIndexer(**kwargs)


def _init_layout(root, **kwargs):
    from .layout import BIDSLayout

    return BIDSLayout(root, **kwargs)
