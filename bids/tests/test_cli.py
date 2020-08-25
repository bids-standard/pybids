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


def test_save_db(runner, tmp_path):
    res = runner.invoke(cli, ['save-db', '--help'])
    assert "Initialize and save an SQLite database index for this BIDS dataset" in res.stdout

    bids_dir = os.path.join(get_test_data_path(), 'ds005')
    db0 = tmp_path / "db0"
    db0.mkdir()
    res = runner.invoke(cli, ['save-db', bids_dir, '--output', str(db0)], catch_exceptions=False)
    assert res.stdout.startswith("Successfully generated database index")
    # rerunning targeting the save directory should fail
    with pytest.raises(RuntimeError):
        runner.invoke(cli, ['save-db', bids_dir, '--output', str(db0)], catch_exceptions=False)

    db1 = tmp_path / "db1"
    db1.mkdir()
    # throw all valid options at it
    res = runner.invoke(
        cli,
        [
            'save-db', bids_dir, '--output', str(db1),
            '--skip-validation', '--skip-metadata',
            '--ignore-path', 'derivatives', '--ignore-path', 'sourcedata',
            '--ignore-regex', r'^\.',
        ],
        catch_exceptions=False,
    )
    assert res.stdout.startswith("Successfully generated database index")
