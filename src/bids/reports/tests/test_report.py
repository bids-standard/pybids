"""Tests for bids.reports.report."""
from collections import Counter

from bids.reports import BIDSReport


def test_report_init(testlayout):
    """Report initialization should return a BIDSReport object."""
    report = BIDSReport(testlayout)
    assert isinstance(report, BIDSReport)


def test_report_gen(testlayout):
    """Report generation should return a counter of unique descriptions in the dataset."""
    report = BIDSReport(testlayout)
    descriptions = report.generate()
    assert isinstance(descriptions, Counter)


def test_report_gen_from_files(testlayout):
    """Report generation from file list should return a counter of unique
    descriptions in the dataset.
    """
    report = BIDSReport(testlayout)
    files = testlayout.get(extension=[".nii.gz", ".nii"])
    descriptions = report.generate_from_files(files)
    assert isinstance(descriptions, Counter)


def test_report_subject(testlayout):
    """Generating a report for one subject should only return one subject's
    description (i.e., one pattern with a count of one).
    """
    report = BIDSReport(testlayout)
    descriptions = report.generate(subject="01")
    assert sum(descriptions.values()) == 1


def test_report_session(testlayout):
    """Generating a report for one session should mean that no other sessions
    appear in any of the unique descriptions.
    """
    report = BIDSReport(testlayout)
    descriptions = report.generate(session="01")
    assert "session 02" not in " ".join(descriptions.keys())


def test_report_file_config(testlayout, config_file):
    """Report initialization should take in a config file and use that if provided."""
    report = BIDSReport(testlayout, config=str(config_file))
    descriptions = report.generate()
    assert isinstance(descriptions, Counter)


def test_report_dict_config(testlayout, testconfig):
    """Report initialization should take in a config dict and use that if provided."""
    report = BIDSReport(testlayout, config=testconfig)
    descriptions = report.generate()
    assert isinstance(descriptions, Counter)
