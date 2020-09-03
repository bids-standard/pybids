import os

from click.testing import CliRunner
import pytest

from bids.cli import cli
from bids.tests import get_test_data_path

@pytest.fixture
def runner():
    return CliRunner()

def test_cli_entrypoint(runner):
    res = runner.invoke(cli, catch_exceptions=False)
    assert "Command-line interface for PyBIDS operations" in res.stdout
    assert runner.invoke(cli, ['-h'], catch_exceptions=False).stdout == res.stdout
    # verify versioning
    assert runner.invoke(cli, ['--version'], catch_exceptions=False).stdout.startswith('pybids')


def test_layout(runner, tmp_path):
    def is_success(res):
        return res.stdout.startswith("Successfully generated database index")

    res = runner.invoke(cli, ['layout', '--help'])
    assert "Initialize a BIDSLayout" in res.stdout

    bids_dir = os.path.join(get_test_data_path(), 'ds005')
    db0 = tmp_path / "db0"
    db0.mkdir()
    res = runner.invoke(cli, ['layout', bids_dir, '--db-path', str(db0)], catch_exceptions=False)
    assert is_success(res)
    # rerunning targeting the save directory should not generate a new index
    res = runner.invoke(cli, ['layout', bids_dir, '--output', str(db0)], catch_exceptions=False)
    assert not is_success(res)
    # but forcing it should
    res = runner.invoke(
        cli, ['layout', bids_dir, '--db-path', str(db0), '--reset-db'], catch_exceptions=False
    )
    assert is_success(res)

    db1 = tmp_path / "db1"
    db1.mkdir()
    # throw the kitchen sink at it
    res = runner.invoke(
        cli,
        [
            'layout', bids_dir, '--db-path', str(db1),
            '--validate', '--no-index-metadata',
            '--ignore', 'derivatives', '--ignore', 'sourcedata', '--ignore', r'/^\./',
            '--force-index', 'test',
            '--config', 'bids',
        ],
        catch_exceptions=False,
    )
    assert is_success(res)
