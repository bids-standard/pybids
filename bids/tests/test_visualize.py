from pathlib import Path

import shutil

import pytest

from bids.layout import BIDSLayout
from bids.visualize import EventPlotter  # type: ignore
from bids.visualize import LayoutPlotter  # type: ignore
from bids.tests import get_test_data_path

"""TODO
-   test on datasets with a large range of events types
"""


@pytest.mark.parametrize(
    "output_dir",
    [
        (Path(__file__).parent.joinpath("tmp")),
        (None),
    ],
)
@pytest.mark.parametrize(
    "show",
    [
        (True),
        (False),
    ],
)
@pytest.mark.parametrize(
    "dataset,filters,plot_by",
    [
        ("ds114", None, "suffix"),
        ("eeg_ds003654s_hed", None, ["suffix", "run"]),
        ("ds000117", dict(session=["mri", "meg"]), None),
        ("fnirs_tapping", None, "suffix"),
    ],
)
def test_LayoutPlotter_smoke(
    dataset, filters, plot_by, show, output_dir, bids_examples
):

    layout = BIDSLayout(root=Path(bids_examples).joinpath(dataset))

    LayoutPlotter(layout, filters=filters).plot(
        plot_by=plot_by, show=show, output_dir=output_dir
    )
    if output_dir is not None:
        shutil.rmtree(output_dir)


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

def test_EventPlotter_too_many_events():

    dataset = Path(get_test_data_path()).joinpath("synthetic")

    file = dataset.joinpath("task-memento_events.tsv")

    with pytest.warns(UserWarning):
        EventPlotter(file)


def test_EventPlotter_warning_event_column(bids_examples):

    with pytest.warns(UserWarning):
        dataset = Path(bids_examples).joinpath("ds001")
        layout = BIDSLayout(dataset)
        files = layout.get(return_type="filename", subject="02", suffix="events")
        EventPlotter(files[0], event_column="foo")


def test_EventPlotter_no_file():
    with pytest.raises(FileNotFoundError):
        EventPlotter("foo.tsv")


def test_EventPlotter_include(bids_examples):

    dataset = Path(bids_examples).joinpath("eeg_ds003654s_hed")
    layout = BIDSLayout(dataset)
    files = layout.get(return_type="filename", subject="002", suffix="events")
    this = EventPlotter(
        files[0], event_column="event_type", include=["show_face", "show_circle"]
    )
    this.plot()
    this.show()


def test_EventPlotter_duration():

    dataset = Path(get_test_data_path()).joinpath("ds000117")
    layout = BIDSLayout(dataset)
    files = layout.get(
        return_type="filename", subject="01", session="mri", suffix="events"
    )
    this = EventPlotter(files[0], event_column="stim_type")
    this.plot()
    this.show()
