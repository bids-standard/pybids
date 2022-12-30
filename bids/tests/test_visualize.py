from pathlib import Path

import pytest

from bids.layout import BIDSLayout
from bids.visualize import EventPlotter  # type: ignore

"""TODO
-   test on datasets with a large range of events types
-   test with limiting event types
"""


@pytest.mark.parametrize(
    "dataset,subject,event_column",
    [
        ("ds001", "02", None),
        ("ds002", "17", None),
        ("ds006", "02", None),
        ("eeg_ds003654s_hed", "002", "event_type"),
    ],
)
def test_EventPlotter_smoke(dataset, subject, event_column, bids_examples):

    dataset = Path(bids_examples).joinpath(dataset)
    layout = BIDSLayout(dataset)

    files = layout.get(return_type="filename", subject=subject, suffix="events")

    this = EventPlotter(files[0], event_column=event_column)
    this.plot()
    this.show()


def test_EventPlotter_only_onset_and_duration_column(bids_examples):

    dataset = Path(bids_examples).joinpath("genetics_ukbb")
    layout = BIDSLayout(dataset)

    files = layout.get(return_type="filename", suffix="events")

    this = EventPlotter(files[0])
    this.plot()
    this.show()


def test_EventPlotter_flag_fast_response(bids_examples):

    dataset = Path(bids_examples).joinpath("ds001")
    layout = BIDSLayout(dataset)

    files = layout.get(return_type="filename", subject="02", suffix="events")

    this = EventPlotter(files[0])
    this.FAST_RESPONSE_THRESHOLD = 0.5
    this.plot()
    this.show()


def test_EventPlotter_warning_event_column(bids_examples):

    with pytest.warns(UserWarning):
        dataset = Path(bids_examples).joinpath("ds001")
        layout = BIDSLayout(dataset)
        files = layout.get(return_type="filename", subject="02", suffix="events")
        this = EventPlotter(files[0], event_column="foo")


def test_EventPlotter_no_file():
    with pytest.raises(FileNotFoundError):
        EventPlotter("foo.tsv")
